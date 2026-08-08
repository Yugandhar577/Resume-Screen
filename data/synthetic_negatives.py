import networkx as nx
import random
from typing import List, Tuple
from models.graph_builder import ResumeGraphBuilder

class SyntheticNegativeGenerator:
    """Generate hard negatives by surgically removing key skills from matching pairs."""
    
    def __init__(self):
        self.graph_builder = ResumeGraphBuilder()
    
    def generate_hard_negative(self, resume_text: str, jd_skills: List[str]) -> str:
        """
        Create hard negative by removing 2-3 key matching skills from resume.
        Forces GAT to learn structural similarity, not just node overlap.
        """
        resume_graph = self.graph_builder.build_graph(resume_text)
        
        # Find skills that match JD requirements
        matching_skills = []
        for node in resume_graph.nodes():
            if any(skill.lower() in node.lower() for skill in jd_skills):
                matching_skills.append(node)
        
        # Remove 2-3 random matching skills to create hard negative
        if len(matching_skills) >= 3:
            skills_to_remove = random.sample(matching_skills, 3)
        elif len(matching_skills) >= 2:
            skills_to_remove = random.sample(matching_skills, 2)
        else:
            return resume_text  # Can't create meaningful negative
        
        # Remove skills from text (simple string replacement)
        modified_text = resume_text
        for skill in skills_to_remove:
            # Remove skill mentions while preserving sentence structure
            modified_text = modified_text.replace(skill, "[REMOVED_SKILL]")
        
        return modified_text
    
    def create_contrastive_pairs(self, resume_text: str, jd_text: str, jd_skills: List[str]) -> List[Tuple[str, str, int]]:
        """
        Returns: [(resume, jd, label)] where label=1 for match, 0 for hard negative
        """
        pairs = []
        
        # Positive pair (original match)
        pairs.append((resume_text, jd_text, 1))
        
        # Hard negative (modified resume)
        hard_negative_resume = self.generate_hard_negative(resume_text, jd_skills)
        if hard_negative_resume != resume_text:
            pairs.append((hard_negative_resume, jd_text, 0))
        
        return pairs