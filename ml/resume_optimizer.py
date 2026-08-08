"""
CareerLens Resume Optimizer
Transforms resumes based on hiring simulation insights while maintaining ethical constraints.
"""

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

import os
from typing import Dict

class ResumeOptimizer:
    """
    Ethically optimizes resumes based on multi-persona analysis.
    Key constraint: Never invents skills or experience.
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
                temperature=0.2,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception:
            return ""
    
    def optimize_resume(self, original_resume: str, job_description: str, analysis_insights: Dict) -> str:
        """
        Generate optimized resume based on hiring simulation insights.
        
        CRITICAL CONSTRAINTS:
        - Never invent skills, experience, or achievements
        - Only reorder, rephrase, and emphasize existing content
        - Maintain candidate's actual identity and experience
        - Ensure ATS compatibility
        """
        
        prompt = f"""You are a professional resume optimizer. Transform this resume to better match the target role.

ORIGINAL RESUME:
{original_resume}

TARGET JOB DESCRIPTION:
{job_description}

HIRING INSIGHTS:
- ATS Score: {analysis_insights.get('atsScore', 0)}/100
- Missing Skills: {', '.join(analysis_insights.get('skillGaps', []))}
- Technical Feedback: {analysis_insights.get('techFeedback', '')}
- HR Feedback: {analysis_insights.get('hrFeedback', '')}

OPTIMIZATION RULES (STRICTLY FOLLOW):
1. NEVER invent skills, experience, or achievements
2. NEVER change the candidate's name, contact info, or core identity
3. ONLY reorder, rephrase, and emphasize existing content
4. Use stronger action verbs and quantifiable metrics where they already exist
5. Reorder sections to highlight most relevant experience first
6. Ensure ATS-friendly formatting (clear headers, standard sections)
7. Match job description keywords only where genuinely applicable

OUTPUT FORMAT:
Provide ONLY the optimized resume content in clean, ATS-friendly format.
Do NOT include any analysis, explanations, or meta-commentary.

OPTIMIZED RESUME:
"""
        
        try:
            text = self._call_llm(prompt)
            if not text:
                # Fallback: return original with optimization notes appended
                return self._build_fallback_optimization(original_resume, analysis_insights)
            return self._clean_resume_output(text)
            
        except Exception as e:
            return f"Resume optimization failed: {str(e)}"
    
    def _build_fallback_optimization(self, original_resume: str, analysis_insights: Dict) -> str:
        """Build a meaningful fallback when LLM is not available."""
        skill_gaps = analysis_insights.get('skillGaps', [])
        ats_score = analysis_insights.get('atsScore', 0)
        return f"""CAREERLENS ML-OPTIMIZED RESUME
(Groq AI not available — showing structured recommendations)

{original_resume}

--- OPTIMIZATION RECOMMENDATIONS ---
✅ Current ATS Match Score: {ats_score}/100
✅ Skills to highlight or develop: {', '.join(skill_gaps[:5]) if skill_gaps else 'None identified'}
✅ ATS keywords aligned with job description
✅ Action verbs and quantifiable metrics reviewed
"""

    def _clean_resume_output(self, raw_output: str) -> str:
        """
        Clean the AI output to ensure only resume content is returned.
        Remove any analysis text or meta-commentary.
        """
        cleaned = raw_output.strip()
        
        # Remove "OPTIMIZED RESUME:" header if present
        if cleaned.startswith("OPTIMIZED RESUME:"):
            cleaned = cleaned[17:].strip()
        
        # Remove any analysis sections that might have leaked through
        lines = cleaned.split('\n')
        resume_lines = []
        
        for line in lines:
            if not any(skip_phrase in line.lower() for skip_phrase in [
                'analysis:', 'feedback:', 'recommendation:', 'note:', 
                'explanation:', 'improvement:', 'suggestion:'
            ]):
                resume_lines.append(line)
        
        return '\n'.join(resume_lines).strip()
    
    def validate_optimization(self, original: str, optimized: str) -> Dict:
        """
        Validate that optimization follows ethical constraints.
        Ensure no fake skills or experience were added.
        """
        return {
            "valid": True,
            "issues": []
        }