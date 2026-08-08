"""
CareerLens Hiring Simulator - Core Intelligence Engine
Simulates three hiring personas: ATS, Technical Lead, HR Recruiter
"""

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

from typing import Dict, List
import re
import os

class HiringSimulator:
    """
    Simulates real hiring pipeline with three distinct personas.
    This is CareerLens's key differentiator - explainable hiring decisions.
    """
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if (_GROQ_AVAILABLE and self.api_key) else None
        self.model_name = "llama3-8b-8192"
    
    def _call_llm(self, prompt: str) -> str:
        """Call Groq LLM or return empty string if unavailable."""
        if not self.client:
            return ""
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            return ""
    
    def simulate_ats(self, resume: str, job_description: str) -> Dict:
        """
        Simulate ATS (Applicant Tracking System) screening.
        Focus: Keyword matching, format compliance, basic requirements.
        """
        
        prompt = f"""
You are an ATS (Applicant Tracking System). Analyze this resume against the job description.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume}

Provide analysis in this EXACT format:
SCORE: [0-100]
KEYWORDS_MATCHED: [list of matched keywords]
MISSING_KEYWORDS: [list of missing critical keywords]
FORMAT_ISSUES: [any formatting problems]
FEEDBACK: [2-3 sentences explaining the score]

Be strict like a real ATS - focus only on keyword matching and basic requirements.
"""
        
        try:
            text = self._call_llm(prompt)
            if not text:
                return {"score": 50, "feedback": "ATS analysis unavailable (no API key)", "missing_skills": []}
            return self._parse_ats_response(text)
        except Exception as e:
            return {
                "score": 0,
                "feedback": f"ATS analysis failed: {str(e)}",
                "missing_skills": []
            }
    
    def simulate_technical_lead(self, resume: str, job_description: str) -> Dict:
        """
        Simulate Technical Lead review.
        Focus: Technical depth, project quality, problem-solving ability.
        """
        
        prompt = f"""
You are a Senior Technical Lead reviewing this resume for a technical role.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume}

Evaluate like a real technical lead would:
- Technical depth and breadth
- Project complexity and impact
- Problem-solving demonstrated
- Technology stack alignment
- Code quality indicators

Provide feedback in this format:
TECHNICAL_ASSESSMENT: [detailed technical evaluation]
STRENGTHS: [technical strengths identified]
CONCERNS: [technical red flags or gaps]
IMPROVEMENTS: [specific technical improvements needed]

Be realistic - technical leads look for depth, not just buzzwords.
"""
        
        try:
            text = self._call_llm(prompt)
            if not text:
                return {"feedback": "Technical analysis unavailable (no API key)", "improvements": []}
            return self._parse_technical_response(text)
        except Exception as e:
            return {
                "feedback": f"Technical analysis failed: {str(e)}",
                "improvements": []
            }
    
    def simulate_hr_recruiter(self, resume: str, job_description: str) -> Dict:
        """
        Simulate HR Recruiter evaluation.
        Focus: Cultural fit, communication, career progression, soft skills.
        """
        
        prompt = f"""
You are an experienced HR Recruiter evaluating this candidate.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume}

Evaluate from HR perspective:
- Career progression and stability
- Communication and presentation
- Cultural fit indicators
- Soft skills demonstration
- Overall candidate appeal

Provide assessment in this format:
HR_EVALUATION: [overall HR assessment]
CAREER_PROGRESSION: [analysis of career growth]
SOFT_SKILLS: [soft skills identified]
CULTURAL_FIT: [cultural fit assessment]
CONCERNS: [any HR red flags]

Think like an HR professional - you care about the whole person, not just technical skills.
"""
        
        try:
            text = self._call_llm(prompt)
            if not text:
                return {"feedback": "HR analysis unavailable (no API key)"}
            return self._parse_hr_response(text)
        except Exception as e:
            return {
                "feedback": f"HR analysis failed: {str(e)}"
            }
    
    def _parse_ats_response(self, response: str) -> Dict:
        """Parse ATS simulation response."""
        try:
            score_match = re.search(r'SCORE:\s*(\d+)', response)
            score = int(score_match.group(1)) if score_match else 0
            
            feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else "No feedback available"
            
            missing_match = re.search(r'MISSING_KEYWORDS:\s*(.+?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)
            missing_skills = missing_match.group(1).strip() if missing_match else ""
            
            return {
                "score": score,
                "feedback": feedback,
                "missing_skills": missing_skills.split(', ') if missing_skills else []
            }
        except Exception:
            return {
                "score": 0,
                "feedback": "Failed to parse ATS response",
                "missing_skills": []
            }
    
    def _parse_technical_response(self, response: str) -> Dict:
        """Parse Technical Lead simulation response."""
        try:
            feedback_match = re.search(r'TECHNICAL_ASSESSMENT:\s*(.+?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else "No technical feedback available"
            
            improvements_match = re.search(r'IMPROVEMENTS:\s*(.+?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)
            improvements = improvements_match.group(1).strip() if improvements_match else ""
            
            return {
                "feedback": feedback,
                "improvements": improvements.split('\n') if improvements else []
            }
        except Exception:
            return {
                "feedback": "Failed to parse technical response",
                "improvements": []
            }
    
    def _parse_hr_response(self, response: str) -> Dict:
        """Parse HR Recruiter simulation response."""
        try:
            feedback_match = re.search(r'HR_EVALUATION:\s*(.+?)(?=\n[A-Z_]+:|$)', response, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else "No HR feedback available"
            
            return {"feedback": feedback}
        except Exception:
            return {"feedback": "Failed to parse HR response"}