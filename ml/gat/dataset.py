"""
GAT Dataset with Synthetic Hard Negatives
Creates contrastive pairs for training the Graph Attention Network.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import random
from ml.graph.builder import KnowledgeGraphBuilder
from ml.graph.converter import GraphConverter

class ContrastiveResumeDataset(Dataset):
    """
    Dataset for contrastive learning with synthetic hard negatives.
    Key innovation: Creates hard negatives by removing key skills from matching pairs.
    """
    
    def __init__(
        self,
        resume_texts: List[str],
        jd_texts: List[str],
        labels: List[int],
        hard_negative_ratio: float = 3.0,
        skill_removal_count: int = 3
    ):
        """
        Args:
            resume_texts: List of resume text strings
            jd_texts: List of job description text strings  
            labels: List of binary labels (1 for match, 0 for no match)
            hard_negative_ratio: Number of hard negatives per positive pair
            skill_removal_count: Number of skills to remove for hard negatives
        """
        self.resume_texts = resume_texts
        self.jd_texts = jd_texts
        self.labels = labels
        self.hard_negative_ratio = hard_negative_ratio
        self.skill_removal_count = skill_removal_count
        
        # Initialize graph builder and converter
        self.graph_builder = KnowledgeGraphBuilder()
        self.graph_converter = GraphConverter()
        
        # Create enhanced dataset with synthetic hard negatives
        self.enhanced_pairs = self._create_enhanced_dataset()
        
    def __len__(self):
        return len(self.enhanced_pairs)
    
    def __getitem__(self, idx):
        """
        Returns:
            resume_data: PyTorch Geometric Data for resume
            jd_data: PyTorch Geometric Data for JD
            label: Binary label (1 for match, 0 for negative)
        """
        resume_text, jd_text, label = self.enhanced_pairs[idx]
        
        # Build graphs
        resume_graph = self.graph_builder.build_graph(resume_text, "resume")
        jd_graph = self.graph_builder.build_graph(jd_text, "job_description")
        
        # Convert to PyTorch Geometric format
        resume_data = self.graph_converter.create_homogeneous_graph(resume_graph)
        jd_data = self.graph_converter.create_homogeneous_graph(jd_graph)
        
        return resume_data, jd_data, torch.tensor(label, dtype=torch.float)
    
    def _create_enhanced_dataset(self) -> List[Tuple[str, str, int]]:
        """
        Create enhanced dataset with synthetic hard negatives.
        This is the key innovation for contrastive training.
        """
        enhanced_pairs = []
        
        for i, (resume_text, jd_text, label) in enumerate(zip(
            self.resume_texts, self.jd_texts, self.labels
        )):
            # Add original pair
            enhanced_pairs.append((resume_text, jd_text, label))
            
            # If positive pair, create hard negatives
            if label == 1:
                hard_negatives = self._create_hard_negatives(
                    resume_text, jd_text, int(self.hard_negative_ratio)
                )
                enhanced_pairs.extend(hard_negatives)
        
        # Shuffle the enhanced dataset
        random.shuffle(enhanced_pairs)
        
        print(f"Enhanced dataset: {len(enhanced_pairs)} pairs "
              f"(original: {len(self.resume_texts)})")
        
        return enhanced_pairs
    
    def _create_hard_negatives(self, resume_text: str, jd_text: str, 
                              count: int) -> List[Tuple[str, str, int]]:
        """
        Create hard negative pairs by removing key skills from resume.
        Forces GAT to learn structural similarity, not just keyword overlap.
        """
        hard_negatives = []
        
        # Extract JD skills for targeting
        jd_graph = self.graph_builder.build_graph(jd_text, "job_description")
        jd_skills = [
            data['name'] for node, data in jd_graph.nodes(data=True)
            if data['type'] == 'SKILL'
        ]
        
        for _ in range(count):
            # Create hard negative by removing skills
            modified_resume = self._remove_key_skills(resume_text, jd_skills)
            hard_negatives.append((modified_resume, jd_text, 0))
        
        return hard_negatives
    
    def _remove_key_skills(self, resume_text: str, jd_skills: List[str]) -> str:
        """
        Remove key matching skills from resume to create hard negative.
        """
        # Build resume graph to identify matching skills
        resume_graph = self.graph_builder.build_graph(resume_text, "resume")
        resume_skills = [
            data['name'] for node, data in resume_graph.nodes(data=True)
            if data['type'] == 'SKILL'
        ]
        
        # Find skills that match between resume and JD
        matching_skills = []
        for resume_skill in resume_skills:
            for jd_skill in jd_skills:
                if (resume_skill.lower() in jd_skill.lower() or 
                    jd_skill.lower() in resume_skill.lower()):
                    matching_skills.append(resume_skill)
                    break
        
        # Remove random subset of matching skills
        if len(matching_skills) >= self.skill_removal_count:
            skills_to_remove = random.sample(matching_skills, self.skill_removal_count)
        else:
            skills_to_remove = matching_skills
        
        # Remove skills from text (simple string replacement)
        modified_text = resume_text
        for skill in skills_to_remove:
            # Replace skill mentions with generic placeholder
            modified_text = modified_text.replace(skill, "[REMOVED_SKILL]")
            # Also try lowercase version
            modified_text = modified_text.replace(skill.lower(), "[removed_skill]")
        
        return modified_text

def create_weak_supervision_pairs(
    resumes_df: pd.DataFrame,
    jds_df: pd.DataFrame,
    num_pairs: int = 10000
) -> Tuple[List[str], List[str], List[int]]:
    """
    Create weakly supervised pairs from Kaggle datasets.
    Positive: Resume category matches JD category
    Negative: Resume category doesn't match JD category
    """
    # Category mappings for weak supervision
    category_mappings = {
        "Data Science": ["data scientist", "machine learning", "analytics", "data analyst"],
        "Web Development": ["frontend", "backend", "full stack", "web developer", "react", "angular"],
        "DevOps": ["devops", "cloud", "infrastructure", "sre", "aws", "docker"],
        "Mobile Development": ["android", "ios", "mobile", "react native", "flutter"],
        "Software Engineering": ["software engineer", "backend", "full stack", "python", "java"],
    }
    
    resume_texts = []
    jd_texts = []
    labels = []
    
    for _ in range(num_pairs):
        # Sample random resume and JD
        resume_row = resumes_df.sample(1).iloc[0]
        jd_row = jds_df.sample(1).iloc[0]
        
        resume_text = resume_row.get('Resume_str', '')
        resume_category = resume_row.get('Category', '')
        jd_text = jd_row.get('description', '')
        jd_title = jd_row.get('title', '')
        
        # Weak labelling: category match = positive
        label = 0  # Default negative
        
        if resume_category in category_mappings:
            keywords = category_mappings[resume_category]
            if any(keyword.lower() in jd_title.lower() for keyword in keywords):
                label = 1  # Positive match
        
        resume_texts.append(resume_text)
        jd_texts.append(jd_text)
        labels.append(label)
    
    return resume_texts, jd_texts, labels

def collate_graph_batch(batch):
    """
    Custom collate function for batching graph data.
    """
    resume_data_list = []
    jd_data_list = []
    labels = []
    
    for resume_data, jd_data, label in batch:
        resume_data_list.append(resume_data)
        jd_data_list.append(jd_data)
        labels.append(label)
    
    # Batch graphs
    resume_batch = Batch.from_data_list(resume_data_list)
    jd_batch = Batch.from_data_list(jd_data_list)
    labels_tensor = torch.stack(labels)
    
    return resume_batch, jd_batch, labels_tensor

# Usage example
if __name__ == "__main__":
    # Load sample data (in practice, load from Kaggle datasets)
    sample_resumes = [
        "Senior Python Developer with 5 years experience in Django and React",
        "Data Scientist with machine learning and TensorFlow experience",
        "DevOps Engineer with AWS and Docker expertise"
    ]
    
    sample_jds = [
        "Looking for Python Developer with Django experience",
        "Data Scientist position requiring ML and Python skills", 
        "DevOps role with cloud and containerization experience"
    ]
    
    sample_labels = [1, 1, 1]  # All positive for this example
    
    # Create dataset
    dataset = ContrastiveResumeDataset(
        resume_texts=sample_resumes,
        jd_texts=sample_jds,
        labels=sample_labels,
        hard_negative_ratio=2.0,
        skill_removal_count=2
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_graph_batch
    )
    
    # Test batch
    for resume_batch, jd_batch, labels in dataloader:
        print(f"Resume batch: {resume_batch.x.shape}")
        print(f"JD batch: {jd_batch.x.shape}")
        print(f"Labels: {labels}")
        break