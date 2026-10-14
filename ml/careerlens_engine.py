"""Deterministic, explainable resume analysis. Research models are not loaded here."""

import re
from typing import Dict, List

from ml.skills import extract_skills, skill_evidence
from ml.resume_optimizer import ResumeOptimizer
from ml.requirements import requirements_check
from ml.assessments import evidence_assessments
from ml.rewrite_workflow import proposals, TEMPLATES


class CareerLensEngine:
    def __init__(self, model_dir: str = "ml/models"):
        self.model_dir = model_dir
        self.ml_available = False
        self.optimizer = ResumeOptimizer()

    def analyze_resume_jd_pair(self, resume_text: str, jd_text: str) -> Dict:
        if not resume_text.strip() or not jd_text.strip():
            raise ValueError("A resume and job description are required.")
        resume_graph = self._build_knowledge_graph(resume_text, "resume")
        jd_graph = self._build_knowledge_graph(jd_text, "jd")
        matching = self._match(resume_graph, jd_graph)
        checks = requirements_check(resume_text, jd_text)
        reviewers = evidence_assessments(resume_text, checks)
        bias = self._detect_bias(jd_text)
        optimization = self.optimizer.prepare_resume(resume_text)
        optimization.update({'proposals': proposals(resume_text), 'sourceResume': resume_text,
                             'templates': [{'id': key, 'label': item['label']} for key, item in TEMPLATES.items()]})
        missing = matching["missing_skills"]
        matched = matching["skill_overlap"]
        score = matching["match_score"]
        ats_feedback = (
            f"{len(matched)} of {matching['job_skill_count']} recognized job skills are present "
            f"({score}% coverage). Extra resume skills do not reduce this score."
            if score is not None else
            "No job skills were recognized in the current vocabulary; a meaningful match score cannot be calculated."
        )
        metric_lines = [line.strip() for line in resume_text.splitlines()
                        if re.search(r"\d+(?:\.\d+)?\s*(?:%|percent|users|customers|requests|projects)", line, re.I)]
        tech_feedback = (
            f"Recognized matching skills: {', '.join(matched) or 'none'}. "
            f"{len(metric_lines)} line(s) contain measurable impact. "
            "Skill mentions alone do not establish technical proficiency."
        )
        sections = [name for name in ("experience", "education", "skills", "projects")
                    if re.search(rf"^\s*{name}\s*:?\s*$", resume_text, re.I | re.M)]
        hr_feedback = (
            f"Recognized resume sections: {', '.join(sections) or 'none'}. "
            "Career progression, personality, and hiring outcomes are not inferred from this check."
        )
        recommendations = []
        if missing:
            recommendations.append("Review these missing job skills: " + ", ".join(missing) +
                                   ". Add them only if you actually have relevant experience; otherwise treat them as learning goals.")
        if not metric_lines:
            recommendations.append("Describe measurable impact where you have supporting facts; do not invent metrics.")
        if not sections:
            recommendations.append("Use clear section headings such as Experience, Skills, Education, and Projects.")
        if score is None:
            recommendations.append("Review the job requirements manually; this vocabulary does not cover all occupations.")
        if not recommendations:
            recommendations.append("Check that each matched skill is supported by a specific project or experience bullet.")
        return {
            "analysis": {"atsScore": score, "atsFeedback": ats_feedback,
                         "techFeedback": tech_feedback, "hrFeedback": hr_feedback,
                         "skillGaps": missing, "recommendations": recommendations},
            "optimizedResume": optimization["resume"], "optimization": optimization,
            "detailed_results": {
                "matching": matching, "bias_detection": bias, "career_trajectory": None,
                "graph_analysis": {"resume_graph": resume_graph, "jd_graph": jd_graph},
                "skill_evidence": skill_evidence(resume_text),
                "requirements": checks, "reviewers": reviewers,
            },
            "method": "Rule_Based", "review_method": "Rule_Based",
            "warnings": ["This is recognized-skill coverage, not an ATS acceptance probability or a trained GAT score."],
        }

    def _extract_skills(self, text: str) -> List[str]:
        return extract_skills(text)

    def _extract_roles(self, text: str) -> List[str]:
        roles = ["Software Engineer", "Data Scientist", "Data Analyst", "Product Manager",
                 "Project Manager", "DevOps Engineer", "Machine Learning Engineer", "Frontend Developer",
                 "Backend Developer", "Full Stack Developer", "Engineering Manager", "Tech Lead",
                 "Designer", "Accountant", "Nurse", "Teacher"]
        return sorted(role for role in roles if re.search(rf"(?<!\w){re.escape(role)}(?!\w)", text, re.I))

    def _build_knowledge_graph(self, text: str, graph_type: str) -> Dict:
        skills = extract_skills(text)
        roles = self._extract_roles(text)
        companies = sorted(set(match.strip() for match in re.findall(r"\s@\s+([^\n(|]+)", text)))
        nodes = [{"id": skill, "type": "skill", "weight": 1} for skill in skills]
        nodes += [{"id": role, "type": "role", "weight": 1} for role in roles]
        existing = {node["id"] for node in nodes}
        nodes += [{"id": company, "type": "company", "weight": 1} for company in companies if company not in existing]
        edges = set()
        for line in text.splitlines():
            entities = extract_skills(line) + self._extract_roles(line)
            entities += [company for company in companies if company in line and company not in entities]
            for index, source in enumerate(entities):
                for target in entities[index + 1:]:
                    edges.add(tuple(sorted((source, target))))
        return {"nodes": nodes,
                "edges": [{"source": s, "target": t, "weight": 1} for s, t in sorted(edges)],
                "type": graph_type, "method": "Recognized entities and same-line co-occurrence"}

    def _match(self, resume_graph: Dict, jd_graph: Dict) -> Dict:
        resume = {n["id"] for n in resume_graph["nodes"] if n["type"] == "skill"}
        jd = {n["id"] for n in jd_graph["nodes"] if n["type"] == "skill"}
        overlap = resume & jd
        union = resume | jd
        return {"match_score": round(100 * len(overlap) / len(jd)) if jd else None,
                "graph_similarity": len(overlap) / len(union) if union else 0,
                "skill_overlap": sorted(overlap), "missing_skills": sorted(jd - resume),
                "job_skill_count": len(jd), "method": "Recognized_Skill_Coverage"}

    def _detect_bias(self, jd_text: str) -> Dict:
        patterns = {
            "age-related wording": ["young and energetic", "digital native", "young candidates"],
            "gender-specific wording": ["male candidates only", "female candidates only"],
            "language restriction": ["native english", "mother tongue", "perfect english"],
            "vague role wording": ["rockstar", "ninja", "guru", "culture fit"],
        }
        flags = [{"type": kind, "phrase": phrase}
                 for kind, phrases in patterns.items() for phrase in phrases
                 if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", jd_text, re.I)]
        return {"bias_detected": bool(flags), "bias_types": sorted({f["type"] for f in flags}),
                "flags": flags, "method": "Phrase screening",
                "note": "These phrases warrant contextual review. This is not a fairness certification or calibrated quality score."}
