"""
Module 3: JD Quality & Bias Detector
    Research BERT architecture and rule votes. Fitted aggregation lives in data/label_aggregation.py.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from typing import List, Dict, Tuple
import re
from datetime import datetime

class JDBiasClassifier(nn.Module):
    """
    BERT-based classifier for detecting bias and quality issues in job descriptions.
    Multi-label classification: Vague, Biased, Salary-Mismatched, Experience-Inflated
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 4):
        super(JDBiasClassifier, self).__init__()
        
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Multi-label classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Label names for interpretability
        self.label_names = ["Vague", "Biased", "Salary_Mismatched", "Experience_Inflated"]
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through BERT and classification head."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Multi-label classification
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Binary cross-entropy for multi-label
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        return {"loss": loss, "logits": logits}

class SnorkelLabeller:
    """
    Legacy rule-vote generator (not a fitted Snorkel model).
    Votes are approximations, not probabilities or reviewed bias labels.
    """
    
    def __init__(self):
        # Technology birth years for experience inflation detection
        self.tech_ages = {
            "react": 2013, "vue": 2014, "angular": 2010, "kubernetes": 2014,
            "docker": 2013, "typescript": 2012, "swift": 2014, "rust": 2010,
            "flutter": 2017, "pytorch": 2016, "tensorflow": 2015, "fastapi": 2018
        }
        
        # Bias patterns
        self.bias_patterns = [
            r"rockstar|ninja|guru|wizard",
            r"young and energetic|fresh graduate",
            r"native english|mother tongue",
            r"culture fit",
            r"aggressive|dominant|competitive"
        ]
        
        # Vague language patterns
        self.vague_patterns = [
            r"excellent communication",
            r"fast-paced environment",
            r"detail-oriented",
            r"self-motivated",
            r"team player",
            r"go-getter"
        ]
    
    def generate_weak_labels(self, jd_texts: List[str]) -> np.ndarray:
        """
        Generate weak labels using labelling functions.
        Returns: [num_samples, num_labels] binary matrix
        """
        labels = []
        
        for jd_text in jd_texts:
            sample_labels = [
                self.lf_vague_requirements(jd_text),
                self.lf_biased_language(jd_text),
                self.lf_salary_mismatch(jd_text),
                self.lf_experience_inflation(jd_text),
            ]
            labels.append(sample_labels)
        
        return np.array(labels)
    
    def lf_experience_inflation(self, jd_text: str) -> int:
        """Detect impossible experience requirements."""
        current_year = datetime.now().year
        
        for tech, birth_year in self.tech_ages.items():
            pattern = rf"(\d+)\+?\s*years?\s*(?:of\s*)?{tech}"
            matches = re.finditer(pattern, jd_text.lower())
            
            for match in matches:
                years_required = int(match.group(1))
                max_possible = current_year - birth_year
                
                if years_required > max_possible:
                    return 1  # Experience inflated
        
        return 0
    
    def lf_biased_language(self, jd_text: str) -> int:
        """Detect biased language patterns."""
        for pattern in self.bias_patterns:
            if re.search(pattern, jd_text.lower()):
                return 1  # Biased language detected
        return 0
    
    def lf_vague_requirements(self, jd_text: str) -> int:
        """Detect vague, non-measurable requirements."""
        vague_count = 0
        for pattern in self.vague_patterns:
            if re.search(pattern, jd_text.lower()):
                vague_count += 1
        
        # If more than 3 vague phrases, flag as vague
        return 1 if vague_count >= 3 else 0
    
    def lf_salary_mismatch(self, jd_text: str) -> int:
        """Abstain: no regional occupation-specific compensation evidence exists."""
        # Skill count is not compensation evidence. This task is unsupported here.
        return -1

class JDQualityScorer:
    """Truthful phrase screening; no arbitrary numerical quality or fairness score."""

    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            from ml.model_registry import artifact_status
            status = artifact_status(model_path, 'job_bias')
            raise ValueError('Learned bias inference is not deployed. ' + status.get('reason', 'Validated phrase attribution is required.'))

    def score_jd(self, jd_text):
        from ml.careerlens_engine import CareerLensEngine
        screening = CareerLensEngine()._detect_bias(jd_text)
        return {'method': 'Phrase screening, not a trained BERT classifier',
                'available_learned_model': False, **screening,
                'suggestions': ['Review flagged wording in context; replace unsupported restrictions with concrete job requirements.'] if screening['flags'] else [],
                'unsupported_checks': ['Salary/role mismatch requires regional compensation data.', 'Fairness cannot be certified from a phrase list.']}


if __name__ == '__main__':
    print(JDQualityScorer().score_jd('Seeking young and energetic candidates'))
