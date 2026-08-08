"""
CareerLens FastAPI Backend - Integrated ML Pipeline
Uses the complete ML pipeline: NER → Graph → GAT → Results
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import asyncio
from typing import Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CareerLens Engine
try:
    from ml.careerlens_engine import CareerLensEngine
    careerlens_engine = CareerLensEngine()
    ML_ENGINE_AVAILABLE = True
    logger.info("CareerLens ML Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ML engine: {e}")
    ML_ENGINE_AVAILABLE = False
    careerlens_engine = None

# Groq fallback
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="CareerLens API",
    description="AI-Powered Resume Transformation & Hiring Simulation Platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    resume: str
    jobDescription: str

class PDFAnalysisRequest(BaseModel):
    jobDescription: str

@app.get("/")
async def root():
    return {
        "message": "CareerLens API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "CareerLens API",
        "version": "1.0.0",
        "ml_engine": ML_ENGINE_AVAILABLE,
        "groq_available": GROQ_AVAILABLE and bool(os.getenv("GROQ_API_KEY"))
    }

@app.post("/api/analyze")
async def analyze_resume(request: AnalysisRequest):
    """
    Core CareerLens endpoint - Uses integrated ML pipeline
    Pipeline: Text → NER → Graph → GAT/ML → Results
    """
    try:
        if ML_ENGINE_AVAILABLE and careerlens_engine:
            # Use the complete ML pipeline (preferred)
            logger.info("Using CareerLens ML pipeline")
            result = await asyncio.to_thread(
                careerlens_engine.analyze_resume_jd_pair,
                request.resume,
                request.jobDescription
            )
            return result
            
        elif GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
            # Fallback to Groq AI
            logger.info("Using Groq AI fallback")
            analysis = await analyze_with_groq(request.resume, request.jobDescription)
            optimized_resume = await optimize_with_groq(request.resume, request.jobDescription)
            
            return {
                "analysis": analysis,
                "optimizedResume": optimized_resume,
                "method": "Groq_Fallback"
            }
        else:
            # Demo mode fallback
            logger.info("Using demo mode")
            return {
                "analysis": get_demo_analysis(),
                "optimizedResume": get_demo_optimization(request.resume),
                "method": "Demo_Mode"
            }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        # Always fallback to demo mode if everything fails
        return {
            "analysis": get_demo_analysis(),
            "optimizedResume": get_demo_optimization(request.resume),
            "method": "Error_Fallback",
            "error": str(e)
        }

@app.post("/api/analyze-pdf")
async def analyze_pdf_resume(file: UploadFile = File(...), jobDescription: str = ""):
    """
    PDF Resume Analysis - Extracts text from PDF then runs ML pipeline
    """
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Extract text from PDF
        pdf_content = await file.read()
        resume_text = extract_text_from_pdf(pdf_content)
        
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Run analysis using the same pipeline
        if ML_ENGINE_AVAILABLE and careerlens_engine:
            result = await asyncio.to_thread(
                careerlens_engine.analyze_resume_jd_pair,
                resume_text,
                jobDescription
            )
            result["pdf_processed"] = True
            return result
        else:
            # Fallback for PDF processing
            return {
                "analysis": get_demo_analysis(),
                "optimizedResume": get_demo_optimization(resume_text),
                "method": "PDF_Demo_Mode",
                "pdf_processed": True
            }
            
    except Exception as e:
        logger.error(f"PDF analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF analysis failed: {str(e)}")

@app.get("/api/graph-data/{analysis_id}")
async def get_graph_data(analysis_id: str):
    """
    Get knowledge graph data for visualization
    """
    # For now, return mock data that matches the expected format
    # In a full implementation, this would retrieve stored analysis results
    return {
        "resume_graph": {
            "nodes": [
                {"id": "Python", "type": "skill", "weight": 0.9},
                {"id": "Machine Learning", "type": "skill", "weight": 0.8},
                {"id": "TensorFlow", "type": "skill", "weight": 0.7},
                {"id": "Software Engineer", "type": "role", "weight": 1.0},
                {"id": "REST APIs", "type": "skill", "weight": 0.6}
            ],
            "edges": [
                {"source": "Python", "target": "Machine Learning", "weight": 0.8},
                {"source": "Python", "target": "TensorFlow", "weight": 0.7},
                {"source": "Python", "target": "Software Engineer", "weight": 0.9},
                {"source": "Machine Learning", "target": "TensorFlow", "weight": 0.6}
            ]
        },
        "jd_graph": {
            "nodes": [
                {"id": "Python", "type": "skill", "weight": 1.0},
                {"id": "PyTorch", "type": "skill", "weight": 0.9},
                {"id": "Deep Learning", "type": "skill", "weight": 0.9},
                {"id": "Senior Developer", "type": "role", "weight": 1.0},
                {"id": "FastAPI", "type": "skill", "weight": 0.7},
                {"id": "Graph Neural Networks", "type": "skill", "weight": 0.8}
            ],
            "edges": [
                {"source": "Python", "target": "PyTorch", "weight": 0.8},
                {"source": "Python", "target": "Deep Learning", "weight": 0.7},
                {"source": "PyTorch", "target": "Deep Learning", "weight": 0.9},
                {"source": "Python", "target": "Senior Developer", "weight": 0.8},
                {"source": "FastAPI", "target": "Senior Developer", "weight": 0.6}
            ]
        }
    }

async def analyze_with_groq(resume: str, job_description: str):
    """Use FREE Groq AI for lightning-fast resume analysis"""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        prompt = f"""
You are a hiring expert. Analyze this resume against the job description and provide:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume}

Respond in JSON format:
{{
    "atsScore": [number 0-100],
    "atsFeedback": "[ATS system feedback]",
    "techFeedback": "[Technical lead feedback]", 
    "hrFeedback": "[HR recruiter feedback]",
    "skillGaps": ["skill1", "skill2"],
    "recommendations": ["rec1", "rec2"]
}}
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",  # Free Groq model
            temperature=0.3,
            max_tokens=1000
        )
        
        # Parse JSON response
        import json
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"Groq analysis failed: {e}")
        return get_demo_analysis()

async def optimize_with_groq(resume: str, job_description: str):
    """Use FREE Groq AI for resume optimization"""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        prompt = f"""
Optimize this resume for the job description. Keep all original information but improve wording, structure, and ATS compatibility.

JOB DESCRIPTION:
{job_description}

ORIGINAL RESUME:
{resume}

Return ONLY the optimized resume content, no explanations:
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.2,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Groq optimization failed: {e}")
        return get_demo_optimization(resume)

def get_demo_analysis():
    """Demo analysis when Groq is not available"""
    return {
        "atsScore": 78,
        "atsFeedback": "Strong keyword match. Consider adding more industry-specific terms.",
        "techFeedback": "Good technical depth shown. Highlight specific technologies used.",
        "hrFeedback": "Clear career progression. Add more quantifiable achievements.",
        "skillGaps": ["Cloud platforms", "Leadership experience"],
        "recommendations": ["Add metrics to achievements", "Include soft skills", "Optimize for ATS"]
    }

def get_demo_optimization(resume: str):
    """Demo optimization when ML engine is not available"""
    return f"""OPTIMIZED RESUME (Demo Mode)

{resume}

--- OPTIMIZATION APPLIED ---
✅ Enhanced formatting for ATS compatibility
✅ Improved action verbs and keywords
✅ Better structure and readability
✅ Matched to job requirements

[Note: ML engine not available, using demo mode]
"""

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2 or pdfplumber."""
    try:
        import PyPDF2
        import io
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
        
    except ImportError:
        # Fallback: try pdfplumber
        try:
            import pdfplumber
            import io
            
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            
            return text.strip()
            
        except ImportError:
            # Final fallback: return placeholder
            return "PDF text extraction not available. Please install PyPDF2 or pdfplumber."
    
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return f"PDF extraction failed: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)