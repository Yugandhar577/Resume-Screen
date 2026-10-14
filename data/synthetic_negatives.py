"""Reproducible synthetic negatives, with canonical skill aliases removed."""
import random
from ml.skills import SKILL_ALIASES, _pattern, extract_skills


class SyntheticNegativeGenerator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def generate_hard_negative(self, resume_text, jd_skills):
        matching = sorted(set(extract_skills(resume_text)) & set(jd_skills))
        selected = self.rng.sample(matching, min(3, len(matching)))
        result = resume_text
        for skill in selected:
            for alias in sorted([skill, *SKILL_ALIASES.get(skill, [])], key=len, reverse=True):
                result = _pattern(alias).sub('[REMOVED_SKILL]', result)
        return result

    def create_contrastive_pairs(self, resume_text, jd_text, jd_skills):
        pairs = [(resume_text, jd_text, 1)]
        negative = self.generate_hard_negative(resume_text, jd_skills)
        if negative != resume_text:
            pairs.append((negative, jd_text, 0))
        return pairs
