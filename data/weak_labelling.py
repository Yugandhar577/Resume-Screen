import pandas as pd
import random
import re
from typing import List, Tuple

class WeakLabeller:
    """Generate positive/negative pairs from Kaggle resume + LinkedIn JD datasets."""
    
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        # Job category mappings for weak supervision
        self.category_mappings = {
            "Data Science": ["data scientist", "machine learning", "analytics"],
            "Web Development": ["frontend", "backend", "full stack", "web developer"],
            "DevOps": ["devops", "cloud", "infrastructure", "sre"],
            "Mobile Development": ["android", "ios", "mobile", "react native"],
            "Software Engineering": ["software engineer", "backend", "full stack"],
        }
    
    def load_datasets(self, resume_path: str, jd_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Kaggle resume dataset and LinkedIn JD dataset."""
        resumes_df = pd.read_csv(resume_path)
        jds_df = pd.read_csv(jd_path)
        return resumes_df, jds_df
    
    def match_category(self, resume_category: str, jd_title: str) -> bool:
        """Check if resume category matches JD using weak supervision rules."""
        for category, keywords in self.category_mappings.items():
            if re.sub(r'[^a-z]', '', str(resume_category).casefold()) == re.sub(r'[^a-z]', '', category.casefold()):
                return any(keyword in str(jd_title).casefold() for keyword in keywords)
        return False
    
    def generate_pairs(self, resumes_df: pd.DataFrame, jds_df: pd.DataFrame, 
                      num_pairs: int = 10000) -> List[Tuple[str, str, int]]:
        """
        Generate positive/negative pairs using weak supervision.
        Returns: [(resume_text, jd_text, label)]
        """
        pairs = []
        if resumes_df.empty or jds_df.empty:
            raise ValueError("Resume and jobs datasets cannot be empty.")
        
        for _ in range(num_pairs):
            # Sample random resume and JD
            resume_row = resumes_df.iloc[self.rng.randrange(len(resumes_df))]
            jd_row = jds_df.iloc[self.rng.randrange(len(jds_df))]
            
            resume_text = resume_row.get('Resume_str', '')
            resume_category = resume_row.get('Category', '')
            jd_text = jd_row.get('description', '')
            jd_title = jd_row.get('title', '')
            if not isinstance(resume_text, str) or not resume_text.strip() or not isinstance(jd_text, str) or not jd_text.strip():
                continue
            
            # Weak labelling: category match = positive, mismatch = negative
            if self.match_category(resume_category, jd_title):
                label = 1  # Positive pair
            else:
                label = 0  # Negative pair
            
            pairs.append((resume_text, jd_text, label))
        
        return pairs
    
    def balance_dataset(self, pairs: List[Tuple[str, str, int]]) -> List[Tuple[str, str, int]]:
        """Ensure 50/50 positive/negative split."""
        positive_pairs = [p for p in pairs if p[2] == 1]
        negative_pairs = [p for p in pairs if p[2] == 0]
        
        min_count = min(len(positive_pairs), len(negative_pairs))
        
        balanced_pairs = (
            self.rng.sample(positive_pairs, min_count) +
            self.rng.sample(negative_pairs, min_count)
        )
        
        self.rng.shuffle(balanced_pairs)
        return balanced_pairs
