"""
Module 1: NER-based Knowledge Graph Builder
Legacy optional spaCy/pattern extractor. The active versioned learned graph uses ml/entity_graph.py.
"""

try:
    import spacy
    _SPACY_AVAILABLE = True
except (ImportError, ValueError, TypeError):
    _SPACY_AVAILABLE = False

from typing import List, Dict, Tuple
import re
from ml.skills import SKILL_ALIASES, _pattern

class CareerNERExtractor:
    """
    Optional custom model or pretrained general NER, supplemented with career patterns.
    Custom entity types: SKILL, ROLE, TOOL, YEAR_EXP, EDUCATION
    Falls back to regex-only mode if spacy is not installed.
    """
    
    def __init__(self, model_path: str = "ml/models/career_ner"):
        self.nlp = None
        if _SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_path)
            except OSError:
                print("Custom NER model not found, trying en_core_web_trf...")
                try:
                    self.nlp = spacy.load("en_core_web_trf")
                except OSError:
                    print("en_core_web_trf not found, using lightweight en_core_web_sm model...")
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        print("en_core_web_sm not found. Running in regex-only mode.")
                        self.nlp = None
        else:
            print("spacy not installed. Running CareerNERExtractor in regex-only mode.")
            
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract career entities from text.
        
        Returns:
            List of entities with type, text, start, end positions
        """
        entities = []

        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, "_.confidence", 1.0)
                })
            
        # Post-process to extract additional patterns (always runs)
        entities.extend(self._extract_experience_years(text))
        entities.extend(self._extract_skills_patterns(text))
        from ml.careerlens_engine import CareerLensEngine
        for role in CareerLensEngine()._extract_roles(text):
            for match in re.finditer(re.escape(role), text, re.I):
                entities.append({"text": role, "label": "ROLE", "start": match.start(), "end": match.end(), "confidence": 1.0})
        
        return entities
    
    def _extract_experience_years(self, text: str) -> List[Dict]:
        """Extract years of experience patterns."""
        patterns = [
            r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)",
            r"(\d+)\+?\s*yrs?\s*(?:experience|exp)?",
            r"(\d+)\+?\s*years?\s*in\s*(\w+)",
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "text": match.group(0),
                    "label": "YEAR_EXP",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9,
                    "years": int(match.group(1))
                })
        
        return entities
    
    def _extract_skills_patterns(self, text: str) -> List[Dict]:
        """Extract common technical skills using patterns."""
        entities = []
        for skill, aliases in SKILL_ALIASES.items():
            spans = set()
            for alias in [skill, *aliases]:
                for match in _pattern(alias).finditer(text):
                    if alias in ("Go", "R") and match.group() != alias:
                        continue
                    if (match.start(), match.end()) in spans:
                        continue
                    spans.add((match.start(), match.end()))
                    entities.append({"text": skill, "label": "SKILL", "start": match.start(),
                                     "end": match.end(), "confidence": 1.0, "category": "VOCABULARY"})
        return entities

    def extract_structured_data(self, text: str) -> Dict:
        """
        Extract and structure all career-relevant information.
        
        Returns:
            Structured dictionary with categorized entities
        """
        entities = self.extract_entities(text)
        
        structured = {
            "skills": [],
            "roles": [],
            "tools": [],
            "education": [],
            "experience_years": [],
            "companies": [],
            "raw_entities": entities
        }
        
        for entity in entities:
            label = entity["label"]
            
            if label == "SKILL":
                structured["skills"].append({
                    "name": entity["text"],
                    "category": entity.get("category", "GENERAL"),
                    "confidence": entity["confidence"]
                })
            elif label == "ROLE":
                structured["roles"].append({
                    "title": entity["text"],
                    "confidence": entity["confidence"]
                })
            elif label == "TOOL" or label == "PRODUCT":
                structured["tools"].append({
                    "name": entity["text"],
                    "confidence": entity["confidence"]
                })
            elif label == "EDUCATION" or (label == "ORG" and re.search(r"university|college|institute|school", entity["text"], re.I)):
                structured["education"].append({
                    "institution": entity["text"],
                    "confidence": entity["confidence"]
                })
            elif label == "YEAR_EXP":
                structured["experience_years"].append({
                    "text": entity["text"],
                    "years": entity.get("years", 0),
                    "confidence": entity["confidence"]
                })
            elif label == "ORG":
                structured["companies"].append({
                    "name": entity["text"],
                    "confidence": entity["confidence"]
                })
        
        return structured

# Usage example
if __name__ == "__main__":
    extractor = CareerNERExtractor()
    
    sample_text = """
    Software Engineer with 5 years of experience in Python, Django, and React.
    Worked at Google and Microsoft. Built REST APIs using FastAPI and deployed
    on AWS. Experience with PostgreSQL and Redis. B.Tech in Computer Science.
    """
    
    entities = extractor.extract_entities(sample_text)
    structured = extractor.extract_structured_data(sample_text)
    
    print("Extracted Entities:")
    for entity in entities:
        print(f"  {entity['label']}: {entity['text']}")
    
    print("\nStructured Data:")
    for key, value in structured.items():
        if value and key != "raw_entities":
            print(f"  {key}: {value}")
