"""CareerLens API: rule-based analysis with optional Groq reviewer feedback."""

import asyncio
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from ml.careerlens_engine import CareerLensEngine
from ml.hiring_simulator import HiringSimulator
from ml.rewrite_workflow import apply_edits
from ml.semantic_matching import SemanticMatcher
from integrations.job_boards import Arbeitnow
from ml.pdf_processing import extract_pdf, ocr_available
from ml.model_registry import artifact_status
from integrations.request_guard import RequestGuard

logger = logging.getLogger(__name__)
careerlens_engine = CareerLensEngine()
semantic_matcher = SemanticMatcher(os.getenv('GAT_MODEL_DIR', 'ml/models/gat_experimental'))
job_board = Arbeitnow()
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

    @field_validator("resume", "jobDescription")
    @classmethod
    def require_content(cls, value):
        if not value.strip():
            raise ValueError("Text cannot be blank.")
        return value.strip()


class RewriteRequest(AnalysisRequest):
    acceptedEdits: list[str] = Field(default_factory=list, max_length=500)
    template: str = Field(default='original', max_length=40)


@app.get("/")
async def root():
    return {"message": "CareerLens API is running", "version": "1.1.0", "status": "healthy"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "CareerLens API", "version": "1.1.0",
            "analysis_mode": "Baseline_Rules_With_Optional_Semantic", "ml_engine": False,
            "groq_available": HiringSimulator.is_configured(), "authentication_required": bool(os.getenv('API_AUTH_TOKEN', '').strip()),
            "semantic_model": semantic_matcher.status(),
            "research_models": {"rewriter": artifact_status('ml/models/t5_rewriter', 'resume_rewriting'),
                                "bias": artifact_status('ml/models/jd_bias', 'job_bias'),
                                "career": artifact_status('ml/models/career', 'career_trajectory')},
            "capabilities": {"text_analysis": True, "pdf_text_extraction": True,
                             "formatting_cleanup": True, "trained_matching": semantic_matcher.status()['available'],
                             "career_prediction": False, "generative_rewriting": False, "reviewed_rewrites": True, "ocr": ocr_available()}}


async def perform_analysis(resume, jd, use_ai=False, use_semantic=False):
    try:
        result = await asyncio.to_thread(careerlens_engine.analyze_resume_jd_pair, resume, jd)
        if use_semantic:
            try:
                result['detailed_results']['semantic_matching'] = await asyncio.to_thread(semantic_matcher.analyze, resume, jd)
            except Exception:
                result['detailed_results']['semantic_matching'] = {'available': False, 'reason': 'Local semantic inference failed; skill coverage remains available.'}
            result['warnings'].append('Semantic matching is experimental and has only been evaluated on synthetic relevance labels; it does not change the baseline score.')
        if use_ai and HiringSimulator.is_configured():
            try:
                feedback = await asyncio.to_thread(HiringSimulator().review, resume, jd)
                result["analysis"].update({key: feedback[key] for key in ('atsFeedback', 'techFeedback', 'hrFeedback')})
                if feedback.get('reviewers'):
                    result['detailed_results']['reviewers'] = feedback['reviewers']
                result["review_method"] = "Groq"
            except Exception:
                logger.warning("Groq review unavailable; returning rule-based results.")
                result["warnings"].append("AI reviewer feedback is unavailable; the results use rule-based checks.")
        elif use_ai:
            result["warnings"].append("Groq is not configured; the results use rule-based checks.")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Resume analysis failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@app.post("/api/analyze")
async def analyze_resume(request: AnalysisRequest):
    return await perform_analysis(request.resume, request.jobDescription, request.useAI, request.useSemantic)


@app.post('/api/optimize-preview')
async def optimize_preview(request: RewriteRequest):
    try:
        optimization = apply_edits(request.resume, request.acceptedEdits, request.template)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    before = await perform_analysis(request.resume, request.jobDescription)
    after = await perform_analysis(optimization['resume'], request.jobDescription)
    original_score, revised_score = before['analysis']['atsScore'], after['analysis']['atsScore']
    after['optimization'] = {**optimization, 'sourceResume': request.resume,
                             'proposals': before['optimization']['proposals'], 'templates': before['optimization']['templates']}
    after['optimizedResume'] = optimization['resume']
    after['comparison'] = {'before': original_score, 'after': revised_score,
                           'delta': revised_score - original_score if original_score is not None and revised_score is not None else None,
                           'metric': 'Recognized skill coverage',
                           'note': 'Factual wording changes may improve clarity without changing skill coverage.'}
    return after


@app.post("/api/analyze-pdf")
async def analyze_pdf_resume(file: UploadFile = File(...), jobDescription: str = Form(...), useAI: bool = Form(False), useSemantic: bool = Form(False), useOCR: bool = Form(False), pdfPassword: str = Form('')):
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
        document = await asyncio.to_thread(extract_pdf, content, useOCR, pdfPassword)
        text = document['text']
        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found. Scanned PDFs need OCR; paste the text instead.")
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=413, detail="Extracted resume text exceeds 50,000 characters.")
        result = await perform_analysis(text.strip(), jobDescription.strip(), useAI, useSemantic)
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
async def search_jobs(query: str = '', limit: int = 10):
    if len(query) > 100 or not 1 <= limit <= 20:
        raise HTTPException(status_code=422, detail='Use a search query of at most 100 characters and a limit of 1–20.')
    try:
        return await asyncio.to_thread(job_board.search, query, limit)
    except Exception:
        raise HTTPException(status_code=502, detail='Job listings are temporarily unavailable. Paste a job description instead.')


def extract_text_from_pdf(content: bytes) -> str:
    """Compatibility helper uses the maintained document workflow."""
    return extract_pdf(content)['text']


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv('BACKEND_HOST', '127.0.0.1'), port=int(os.getenv("BACKEND_PORT", "8001")))
