import openai
import pandas as pd
from typing import List, Tuple
import time

class GPTDistillationLabeller:
    """Use GPT-3.5 to generate improved resume bullets for T5 training data."""
    
    def __init__(self, api_key: str):
        openai.api_key = api_key
    
    def improve_bullet(self, original_bullet: str) -> str:
        """Use GPT-3.5 to improve a single resume bullet point."""
        prompt = f"""
Rewrite this resume bullet point to be more impactful and ATS-friendly:

Original: {original_bullet}

Make it:
- Start with strong action verb
- Include quantifiable metrics where possible
- Use industry keywords
- Be concise but specific

Improved:"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            
            improved = response.choices[0].message.content.strip()
            return improved
            
        except Exception as e:
            print(f"Error improving bullet: {e}")
            return original_bullet
    
    def generate_training_pairs(self, resume_bullets: List[str], 
                              batch_size: int = 50) -> List[Tuple[str, str]]:
        """
        Generate improved versions of resume bullets for T5 training.
        Cost: ~$3 for 1000 bullets with GPT-3.5-turbo
        """
        pairs = []
        
        for i in range(0, len(resume_bullets), batch_size):
            batch = resume_bullets[i:i + batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{len(resume_bullets)//batch_size + 1}")
            
            for bullet in batch:
                if len(bullet.strip()) > 10:  # Skip very short bullets
                    improved = self.improve_bullet(bullet)
                    pairs.append((bullet, improved))
                    
                    # Rate limiting to avoid API limits
                    time.sleep(0.1)
        
        return pairs
    
    def extract_bullets_from_resumes(self, resumes_df: pd.DataFrame) -> List[str]:
        """Extract individual bullet points from resume dataset."""
        bullets = []
        
        for _, row in resumes_df.iterrows():
            resume_text = row.get('Resume_str', '')
            
            # Split by common bullet indicators
            lines = resume_text.split('\n')
            for line in lines:
                line = line.strip()
                # Look for bullet-like patterns
                if (line.startswith('•') or line.startswith('-') or 
                    line.startswith('*') or 
                    (len(line) > 20 and len(line) < 200)):  # Reasonable bullet length
                    
                    clean_bullet = line.lstrip('•-*').strip()
                    if len(clean_bullet) > 15:
                        bullets.append(clean_bullet)
        
        return bullets[:1000]  # Limit to 1000 for cost control

# Usage example
def create_t5_training_data():
    """
    Generate T5 training pairs using GPT distillation.
    Total cost: ~$3 for 1000 bullet improvements.
    """
    # Load resume dataset
    # resumes_df = pd.read_csv('kaggle_resumes.csv')
    
    # Extract bullets
    # labeller = GPTDistillationLabeller(api_key="your_openai_key")
    # bullets = labeller.extract_bullets_from_resumes(resumes_df)
    
    # Generate improvements (costs ~$3)
    # pairs = labeller.generate_training_pairs(bullets)
    
    # Save for T5 training
    # pairs_df = pd.DataFrame(pairs, columns=['original', 'improved'])
    # pairs_df.to_csv('t5_training_pairs.csv', index=False)
    
    pass