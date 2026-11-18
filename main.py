"""CareerLens API: rule-based analysis with optional local Gemma review."""

import asyncio
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator, HttpUrl, model_validator, ConfigDict
from typing import Literal

load_dotenv()

from ml.careerlens_engine import CareerLensEngine
from ml.rewrite_workflow import apply_edits
from ml.local_rewriter import generate_proposals, apply_local_edits, read_token
from ml.local_review import review_locally
from integrations.local_llm import local_model, LocalModelError
from ml.semantic_matching import SemanticMatcher
from integrations.job_boards import Arbeitnow, RemoteOK
from ml.pdf_processing import extract_pdf, extract_pdf_isolated, ocr_available
from ml.model_registry import artifact_status
from integrations.request_guard import RequestGuard
from ml.pdf_export import create_pdf
from ml.career_matching import status as career_matching_status
from ml.evidence_matching import status as evidence_matching_status

logger = logging.getLogger(__name__)
careerlens_engine = CareerLensEngine()
semantic_matcher = SemanticMatcher(os.getenv('GAT_MODEL_DIR', 'ml/models/gat_entities'))
job_board = Arbeitnow()
remote_job_board = RemoteOK()
MAX_TEXT_LENGTH = 50000
MAX_PDF_BYTES = 5 * 1024 * 1024
app = FastAPI(title="CareerLens API", version="1.1.0")
if os.getenv('API_AUTH_TOKEN', '') and len(os.getenv('API_AUTH_TOKEN', '').strip()) < 32:
    raise ValueError('API_AUTH_TOKEN must contain at least 32 characters, or be unset for local development.')
app.add_middleware(RequestGuard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if origin.strip()],
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"],
)


class AnalysisRequest(BaseModel):
    resume: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    jobDescription: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    useAI: bool = False
    useSemantic: bool = False
    useResearch: bool = False
    useEvidence:bool=False

    @field_validator("resume", "jobDescription")
    @classmethod
    def require_content(cls, value):
        if not value.strip():
            raise ValueError("Text cannot be blank.")
        return value.strip()


class RewriteRequest(AnalysisRequest):
    acceptedEdits: list[str] = Field(default_factory=list, max_length=500)
    template: str = Field(default='original', max_length=40)
    proposalToken: str = Field(default='', max_length=50000)


class ProposalRequest(AnalysisRequest):
    engine: Literal['gemma', 't5'] = 'gemma'


class CareerMatchRequest(BaseModel):
    resume: str = Field(min_length=1,max_length=MAX_TEXT_LENGTH)
    currentRoleCode: str = Field(default='',max_length=40)
    targetRoleCodes: list[str] = Field(default_factory=list,max_length=3)


class SalaryReference(BaseModel):
    model_config=ConfigDict(extra='forbid')
    id:str=Field(default='user-reference',max_length=80)
    title:str=Field(min_length=1,max_length=200)
    role:str=Field(min_length=1,max_length=150)
    url:HttpUrl
    as_of:str=Field(pattern=r'^20\d{2}(?:-\d{2}(?:-\d{2})?)?$')
    country:Literal['India']='India'
    currency:Literal['INR']='INR'
    pay_basis:Literal['ctc','base']
    geography:str=Field(min_length=1,max_length=200)
    method:str=Field(min_length=1,max_length=500)
    license:str=Field(min_length=3,max_length=500)
    experience_min:float=Field(ge=0,le=60,allow_inf_nan=False)
    experience_max:float=Field(ge=0,le=60,allow_inf_nan=False)
    annual_low:float=Field(gt=0,lt=1e11,allow_inf_nan=False)
    annual_high:float=Field(gt=0,lt=1e11,allow_inf_nan=False)

    @model_validator(mode='after')
    def ordered(self):
        from datetime import date
        if self.annual_high<self.annual_low or self.experience_max<self.experience_min:
            raise ValueError('Benchmark ranges must be ordered.')
        try:
            stamp=date.fromisoformat(self.as_of+('-01-01' if len(self.as_of)==4 else '-01' if len(self.as_of)==7 else ''))
        except ValueError:raise ValueError('Invalid source date.')
        if stamp>date.today():raise ValueError('Source date cannot be in the future.')
        return self


class SalaryRequest(BaseModel):
    offerText: str = Field(min_length=1,max_length=MAX_TEXT_LENGTH)
    referenceId: str = Field(default='',max_length=80)
    reference:SalaryReference|None=None
    experienceYears: float = Field(ge=0,le=60,allow_inf_nan=False)
    city: str = Field(default='India',max_length=100)
    payBasis: Literal['ctc','base','unspecified'] = 'unspecified'


class ExportRequest(BaseModel):
    resume: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator('resume')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Resume text cannot be blank.')
        return value


@app.get("/")
async def root():
    return {"message": "CareerLens API is running", "version": "1.1.0", "status": "healthy"}


@app.get("/health")
async def health_check():
    local_status = await asyncio.to_thread(local_model.status)
    return {"status": "healthy", "service": "CareerLens API", "version": "1.1.0",
            "analysis_mode": "Evidence_Rules_With_Local_Gemma_And_Optional_GAT", "ml_engine": False,
            "groq_available": False, "local_ai": local_status, "authentication_required": bool(os.getenv('API_AUTH_TOKEN', '').strip()),
            "semantic_model": semantic_matcher.status(),
            "research_models": {"rewriter": artifact_status('ml/models/t5_rewriter', 'resume_rewriting'),
                                "bias": artifact_status('ml/models/jd_bias', 'job_bias'),
                                "weak_labels": artifact_status('ml/models/weak_labels', 'job_weak_labels'),
                                "career": artifact_status('ml/models/career', 'career_trajectory')},
            "capabilities": {"text_analysis": True, "pdf_text_extraction": True,
                             "formatting_cleanup": True, "trained_matching": semantic_matcher.status()['available'],
                             "career_prediction": False, "generative_rewriting": local_status['available'], "local_rewrite_proposals": local_status['available'],
                             "local_evidence_review": local_status['available'], "reviewed_rewrites": True, "salary_comparison": True, "semantic_evidence_retrieval":evidence_matching_status()['available'], "skill_based_career_matching": career_matching_status()['available'], "ocr": ocr_available()}}


async def perform_analysis(resume, jd, use_ai=False, use_semantic=False, use_research=False,use_evidence=False):
    try:
        result = await asyncio.to_thread(careerlens_engine.analyze_resume_jd_pair, resume, jd)
        if use_evidence:
            from ml.evidence_matching import analyze
            try:
                result['detailed_results']['evidence_matching']=await asyncio.to_thread(analyze,resume,jd)
            except Exception:
                result['detailed_results']['evidence_matching']={'available':False,'reason':'Local semantic passage retrieval failed. Baseline evidence checks remain available.'}
        if use_research:
            from ml.research_inference import jd_experiment
            try:
                result['detailed_results']['jd_experiment'] = await asyncio.to_thread(jd_experiment, jd)
            except Exception:
                result['detailed_results']['jd_experiment'] = {'available': False, 'reason': 'Local BERT experiment failed; rule and Gemma reviews remain separate.'}
        if use_semantic:
            try:
                result['detailed_results']['semantic_matching'] = await asyncio.to_thread(semantic_matcher.analyze, resume, jd)
            except Exception:
                result['detailed_results']['semantic_matching'] = {'available': False, 'reason': 'Local semantic inference failed; skill coverage remains available.'}
            result['warnings'].append('Semantic matching is experimental and has only been evaluated on synthetic relevance labels; it does not change the baseline score.')
        if use_ai:
            try:
                report = await asyncio.to_thread(review_locally, resume, jd)
                result['detailed_results']['local_review'] = report
                result['detailed_results']['reviewers'] = report['reviewers']
                fields = {'ATS & structure': 'atsFeedback', 'Technical evidence': 'techFeedback', 'Recruiter clarity': 'hrFeedback'}
                for reviewer in report['reviewers']:
                    notes = [f['statement'] for f in reviewer['strengths']+reviewer['concerns']]+reviewer['recommendations']
                    result['analysis'][fields[reviewer['persona']]] = ' '.join(notes) or 'No supported local findings were returned; review the evidence checklist.'
                result["review_method"] = "Local_Ollama"
                if report['rejected_unverified_items']:
                    result['warnings'].append(f"Discarded {report['rejected_unverified_items']} local AI items whose source evidence could not be verified.")
            except Exception:
                logger.warning("Local reviewer unavailable; returning rule-based results.")
                result['detailed_results']['local_review'] = {'available': False, 'reason': 'Local AI could not complete this review. Check Ollama, model availability, and input length, then retry.'}
                result["warnings"].append("Local AI review is unavailable; the results use rule-based checks. No cloud fallback was used.")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Resume analysis failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@app.post("/api/analyze")
async def analyze_resume(request: AnalysisRequest):
    return await perform_analysis(request.resume, request.jobDescription, request.useAI, request.useSemantic, request.useResearch,request.useEvidence)


@app.post('/api/optimize-preview')
async def optimize_preview(request: RewriteRequest):
    try:
        if request.proposalToken:
            optimization = apply_local_edits(request.resume, request.jobDescription, request.acceptedEdits, request.template, request.proposalToken)
        else:
            optimization = apply_edits(request.resume, request.acceptedEdits, request.template)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    before = await perform_analysis(request.resume, request.jobDescription)
    after = await perform_analysis(optimization['resume'], request.jobDescription)
    original_score, revised_score = before['analysis']['atsScore'], after['analysis']['atsScore']
    after['optimization'] = {**optimization, 'sourceResume': request.resume,
                             'proposals': before['optimization']['proposals'], 'templates': before['optimization']['templates']}
    if request.proposalToken:
        local_edits = read_token(request.proposalToken, request.resume, request.jobDescription)
        local_lines = {e['line'] for e in local_edits}
        after['optimization']['proposals'] = [e for e in before['optimization']['proposals'] if e['line'] not in local_lines]+local_edits
        after['optimization']['proposalToken'] = request.proposalToken
    after['optimizedResume'] = optimization['resume']
    after['comparison'] = {'before': original_score, 'after': revised_score,
                           'delta': revised_score - original_score if original_score is not None and revised_score is not None else None,
                           'metric': 'Recognized skill coverage',
                           'note': 'Factual wording changes may improve clarity without changing skill coverage.'}
    return after


@app.post('/api/rewrite-proposals')
async def local_rewrite_proposals(request: ProposalRequest):
    try:
        return await asyncio.to_thread(generate_proposals, request.resume, request.jobDescription, request.engine)
    except LocalModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=502, detail='Local rewrite generation or validation failed. Existing proposals remain available.')


@app.post('/api/export-pdf')
async def export_resume_pdf(request: ExportRequest):
    try:
        content = await asyncio.to_thread(create_pdf, request.resume)
        return Response(content, media_type='application/pdf', headers={'Content-Disposition':'attachment; filename="careerlens-resume.pdf"'})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/analyze-pdf")
async def analyze_pdf_resume(file: UploadFile = File(...), jobDescription: str = Form(...), useAI: bool = Form(False), useSemantic: bool = Form(False), useResearch: bool = Form(False), useEvidence:bool=Form(False), useOCR: bool = Form(False), pdfPassword: str = Form('')):
    try:
        if not jobDescription.strip() or len(jobDescription) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=422, detail="Enter a job description of 1–50,000 characters.")
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        content = await file.read(MAX_PDF_BYTES + 1)
        if len(content) > MAX_PDF_BYTES:
            raise HTTPException(status_code=413, detail="PDF files must be 5 MB or smaller.")
        if not content.lstrip().startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")
        if len(pdfPassword) > 200:
            raise HTTPException(status_code=422, detail='PDF password is too long.')
        document = await asyncio.to_thread(extract_pdf_isolated, content, useOCR, pdfPassword)
        text = document['text']
        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found. Scanned PDFs need OCR; paste the text instead.")
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=413, detail="Extracted resume text exceeds 50,000 characters.")
        result = await perform_analysis(text.strip(), jobDescription.strip(), useAI, useSemantic, useResearch,useEvidence)
        result["pdf_processed"] = True
        result['document'] = {key: value for key, value in document.items() if key != 'text'}
        result['warnings'].extend(document['warnings'])
        return result
    finally:
        await file.close()


@app.get("/api/graph-data/{analysis_id}")
async def get_graph_data(analysis_id: str):
    raise HTTPException(status_code=410, detail="Use detailed_results.graph_analysis from your analysis response.")


@app.get('/api/jobs')
async def search_jobs(query: str = '', limit: int = 10, source: str = 'arbeitnow'):
    if len(query) > 100 or not 1 <= limit <= 20:
        raise HTTPException(status_code=422, detail='Use a search query of at most 100 characters and a limit of 1–20.')
    if source not in ('arbeitnow','remoteok'):
        raise HTTPException(status_code=422, detail='Choose Arbeitnow or Remote OK.')
    try:
        return await asyncio.to_thread((job_board if source == 'arbeitnow' else remote_job_board).search, query, limit)
    except Exception:
        raise HTTPException(status_code=502, detail='Job listings are temporarily unavailable. Paste a job description instead.')


@app.get('/api/career-roles')
async def career_roles(query: str = ''):
    if not 2 <= len(query.strip()) <= 100:
        raise HTTPException(status_code=422, detail='Enter a role search of 2–100 characters.')
    from ml.career_experiment import search_roles
    try:
        return {'roles':await asyncio.to_thread(search_roles, query.strip())}
    except Exception:
        raise HTTPException(status_code=503, detail='Local ESCO catalog unavailable. Run the explicit taxonomy acquisition step.')


@app.get('/api/salary-references')
async def salary_references():
    from ml.salary import catalog
    return {'country':'India','references':catalog()}


@app.post('/api/salary-compare')
async def salary_compare(request:SalaryRequest):
    from ml.salary import compare
    try:
        reference={**request.reference.model_dump(mode='json'),'user_provided':True,'granularity':'user-provided'} if request.reference else None
        return await asyncio.to_thread(compare,request.offerText,request.referenceId,request.experienceYears,request.city,request.payBasis,reference)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc))


@app.post('/api/career-recommendations')
async def career_recommendations(request:CareerMatchRequest):
    from ml.career_matching import recommend
    try:
        return await asyncio.to_thread(recommend,request.resume,request.currentRoleCode,request.targetRoleCodes)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503,detail='Local career profiles or encoder unavailable. Acquire career profiles and build the career index.')


@app.get('/api/career-explore')
async def career_explore(code: str):
    if len(code)>40:
        raise HTTPException(status_code=422, detail='Invalid role code.')
    from ml.career_experiment import explore
    try:
        return await asyncio.to_thread(explore, code)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail='Local career research artifact or catalog unavailable.')


def extract_text_from_pdf(content: bytes) -> str:
    """Compatibility helper uses the maintained document workflow."""
    return extract_pdf(content)['text']


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv('BACKEND_HOST', '127.0.0.1'), port=int(os.getenv("BACKEND_PORT", "8001")))
