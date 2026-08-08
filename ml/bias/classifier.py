"""
Module 3: JD Quality & Bias Detector
BERT-based multi-label classification with Snorkel weak supervision.
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
    Weak supervision using Snorkel labelling functions.
    Generates probabilistic labels without manual annotation.
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
            r"culture fit|team player",
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
                self.lf_experience_inflation(jd_text),
                self.lf_biased_language(jd_text),
                self.lf_vague_requirements(jd_text),
                self.lf_salary_mismatch(jd_text)
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
        """Detect salary-skill mismatch (simplified)."""
        # Count required skills
        skill_keywords = [
            "python", "java", "javascript", "react", "angular", "aws", "docker",
            "kubernetes", "machine learning", "data science", "sql", "nosql"
        ]
        
        skill_count = sum(1 for skill in skill_keywords if skill in jd_text.lower())
        
        # Check for entry-level indicators
        entry_level_indicators = ["entry level", "junior", "0-2 years", "fresh graduate"]
        is_entry_level = any(indicator in jd_text.lower() for indicator in entry_level_indicators)
        
        # If entry-level but requires many skills, flag as mismatch
        if is_entry_level and skill_count > 8:
            return 1
        
        return 0

class JDQualityScorer:
    """
    Complete JD quality scoring system with explanations.
    """
    
    def __init__(self, model_path: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        if model_path and torch.cuda.is_available():
            # Load trained model
            self.model = JDBiasClassifier()
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
            # Use rule-based fallback
            self.model = None
        
        self.snorkel_labeller = SnorkelLabeller()
    
    def score_jd(self, jd_text: str) -> Dict:
        """
        Score a job description for quality and bias.
        
        Returns:
            Dictionary with scores, flags, and explanations
        """
        if self.model:
            # Use trained BERT model
            ml_scores = self._get_ml_scores(jd_text)
        else:
            # Use rule-based scoring
            ml_scores = self._get_rule_based_scores(jd_text)
        
        # Get detailed explanations
        explanations = self._get_explanations(jd_text)
        
        # Calculate overall quality score (0-100)
        quality_score = self._calculate_quality_score(ml_scores)
        
        return {
            "quality_score": quality_score,
            "flags": {
                "vague": ml_scores[0] > 0.5,
                "biased": ml_scores[1] > 0.5,
                "salary_mismatched": ml_scores[2] > 0.5,
                "experience_inflated": ml_scores[3] > 0.5
            },
            "scores": {
                "vague": ml_scores[0],
                "biased": ml_scores[1], 
                "salary_mismatched": ml_scores[2],
                "experience_inflated": ml_scores[3]
            },
            "explanations": explanations,
            "suggestions": self._get_suggestions(ml_scores, explanations)
        }
    
    def _get_ml_scores(self, jd_text: str) -> List[float]:
        """Get scores from trained BERT model."""
        # Tokenize
        inputs = self.tokenizer(
            jd_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            probabilities = torch.sigmoid(logits).squeeze().tolist()
        
        return probabilities
    
    def _get_rule_based_scores(self, jd_text: str) -> List[float]:
        """Get scores using rule-based labelling functions."""
        labels = self.snorkel_labeller.generate_weak_labels([jd_text])[0]
        # Convert binary labels to probabilities (0.8 if flagged, 0.2 if not)
        scores = [0.8 if label == 1 else 0.2 for label in labels]
        return scores
    
    def _get_explanations(self, jd_text: str) -> Dict:
        """Get detailed explanations for each flag."""
        explanations = {}
        
        # Experience inflation explanations
        current_year = datetime.now().year
        exp_issues = []
        
        for tech, birth_year in self.snorkel_labeller.tech_ages.items():
            pattern = rf"(\d+)\+?\s*years?\s*(?:of\s*)?{tech}"
            matches = re.finditer(pattern, jd_text.lower())
            
            for match in matches:
                years_required = int(match.group(1))
                max_possible = current_year - birth_year
                
                if years_required > max_possible:
                    exp_issues.append(
                        f"Requires {years_required} years of {tech.title()}, "
                        f"but {tech.title()} only exists since {birth_year} "
                        f"({max_possible} years max)"
                    )
        
        explanations["experience_inflated"] = exp_issues
        
        # Bias explanations
        bias_issues = []
        for pattern in self.snorkel_labeller.bias_patterns:
            matches = re.finditer(pattern, jd_text.lower())
            for match in matches:
                bias_issues.append(f"Biased language: '{match.group(0)}'")
        
        explanations["biased"] = bias_issues
        
        # Vague language explanations
        vague_issues = []
        for pattern in self.snorkel_labeller.vague_patterns:
            if re.search(pattern, jd_text.lower()):
                vague_issues.append(f"Vague requirement: '{pattern}'")
        
        explanations["vague"] = vague_issues
        
        return explanations
    
    def _calculate_quality_score(self, scores: List[float]) -> int:
        """Calculate overall quality score (0-100)."""
        # Invert scores (lower bias/vague = higher quality)
        quality_components = [1 - score for score in scores]
        overall_quality = np.mean(quality_components)
        return int(overall_quality * 100)
    
    def _get_suggestions(self, scores: List[float], explanations: Dict) -> List[str]:
        """Get improvement suggestions."""
        suggestions = []
        
        if scores[0] > 0.5:  # Vague
            suggestions.append("Replace vague terms with specific, measurable requirements")
        
        if scores[1] > 0.5:  # Biased
            suggestions.append("Remove exclusionary language and use inclusive terminology")
        
        if scores[2] > 0.5:  # Salary mismatched
            suggestions.append("Align skill requirements with seniority level and compensation")
        
        if scores[3] > 0.5:  # Experience inflated
            suggestions.append("Verify experience requirements are realistic for technology age")
        
        return suggestions

# Usage example
if __name__ == "__main__":
    # Sample problematic job description
    problematic_jd = """
    We're looking for a rockstar React developer with 10+ years of React experience.
    Must be young and energetic with excellent communication skills.
    Looking for a ninja who can work in our fast-paced environment.
    Entry level position requiring expertise in React, Angular, Vue, Python, 
    Java, AWS, Docker, Kubernetes, machine learning, and data science.
    """
    
    # Initialize scorer
    scorer = JDQualityScorer()
    
    # Score the JD
    results = scorer.score_jd(problematic_jd)
    
    print(f"Quality Score: {results['quality_score']}/100")
    print(f"Flags: {results['flags']}")
    print(f"Explanations: {results['explanations']}")
    print(f"Suggestions: {results['suggestions']}")
    
    # Test Snorkel labeller
    labeller = SnorkelLabeller()
    labels = labeller.generate_weak_labels([problematic_jd])
    print(f"Weak labels: {labels[0]}")  # [vague, biased, salary_mismatch, exp_inflated]