"""
Module 4b: Targeted Resume Rewriter
T5 seq2seq model for ATS-optimized resume bullet rewriting.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Dict, Tuple
import re

class ResumeRewriteDataset(Dataset):
    """Dataset for T5 fine-tuning on resume bullet improvements."""
    
    def __init__(self, pairs_df: pd.DataFrame, tokenizer: T5Tokenizer, max_length: int = 512):
        self.pairs = pairs_df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        # Format as T5 task with JD context
        input_text = f"rewrite resume bullet for job: {row.get('jd_context', '')} original: {row['original']}"
        target_text = row['improved']
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

class T5ResumeRewriter:
    """
    T5-based resume bullet rewriter with JD-specific optimization.
    Fine-tuned on distilled GPT data + Reddit improvements.
    """
    
    def __init__(self, model_name: str = "t5-small"):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add special tokens for resume rewriting
        special_tokens = ["<SKILL>", "<METRIC>", "<ACTION>", "<RESULT>"]
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def train(self, training_pairs: pd.DataFrame, save_path: str = "ml/models/t5_rewriter"):
        """
        Fine-tune T5 on resume improvement pairs.
        
        Args:
            training_pairs: DataFrame with 'original', 'improved', 'jd_context' columns
        """
        # Create dataset
        dataset = ResumeRewriteDataset(training_pairs, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=save_path,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{save_path}/logs',
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting T5 fine-tuning...")
        trainer.train()
        
        # Save
        self.model.save_pretrained(f'{save_path}/final')
        self.tokenizer.save_pretrained(f'{save_path}/final')
        
        print(f"Model saved to {save_path}/final")
    
    def rewrite_bullet(self, original_bullet: str, jd_context: str = "", 
                      num_alternatives: int = 3) -> List[Dict]:
        """
        Rewrite a resume bullet point with JD-specific optimization.
        
        Args:
            original_bullet: Original resume bullet
            jd_context: Job description context for targeted rewriting
            num_alternatives: Number of rewrite alternatives
            
        Returns:
            List of rewrite alternatives with scores
        """
        # Format input
        input_text = f"rewrite resume bullet for job: {jd_context} original: {original_bullet}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate alternatives
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                num_return_sequences=num_alternatives,
                num_beams=num_alternatives * 2,
                temperature=0.8,
                do_sample=True,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode alternatives
        alternatives = []
        for i, output in enumerate(outputs):
            rewritten = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Calculate improvement score
            score = self._calculate_improvement_score(original_bullet, rewritten, jd_context)
            
            alternatives.append({
                "rewritten": rewritten,
                "improvement_score": score,
                "changes": self._identify_changes(original_bullet, rewritten)
            })
        
        # Sort by improvement score
        alternatives.sort(key=lambda x: x["improvement_score"], reverse=True)
        
        return alternatives
    
    def _calculate_improvement_score(self, original: str, rewritten: str, jd_context: str) -> float:
        """Calculate improvement score based on ATS optimization factors."""
        score = 0.0
        
        # Action verb improvement
        action_verbs = [
            "achieved", "developed", "implemented", "optimized", "increased",
            "reduced", "built", "designed", "led", "managed", "created"
        ]
        
        original_actions = sum(1 for verb in action_verbs if verb.lower() in original.lower())
        rewritten_actions = sum(1 for verb in action_verbs if verb.lower() in rewritten.lower())
        
        if rewritten_actions > original_actions:
            score += 0.3
        
        # Quantification improvement
        original_numbers = len(re.findall(r'\d+', original))
        rewritten_numbers = len(re.findall(r'\d+', rewritten))
        
        if rewritten_numbers > original_numbers:
            score += 0.3
        
        # JD keyword alignment
        if jd_context:
            jd_keywords = self._extract_keywords(jd_context)
            original_matches = sum(1 for kw in jd_keywords if kw.lower() in original.lower())
            rewritten_matches = sum(1 for kw in jd_keywords if kw.lower() in rewritten.lower())
            
            if rewritten_matches > original_matches:
                score += 0.4
        
        # Length optimization (not too short, not too long)
        optimal_length = 80  # characters
        original_length_score = 1 - abs(len(original) - optimal_length) / optimal_length
        rewritten_length_score = 1 - abs(len(rewritten) - optimal_length) / optimal_length
        
        if rewritten_length_score > original_length_score:
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _extract_keywords(self, jd_text: str) -> List[str]:
        """Extract important keywords from job description."""
        # Simple keyword extraction (in practice, use more sophisticated NLP)
        common_keywords = [
            "python", "java", "javascript", "react", "angular", "vue",
            "aws", "docker", "kubernetes", "sql", "nosql", "api",
            "machine learning", "data science", "backend", "frontend",
            "agile", "scrum", "git", "ci/cd", "testing", "debugging"
        ]
        
        found_keywords = []
        jd_lower = jd_text.lower()
        
        for keyword in common_keywords:
            if keyword in jd_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _identify_changes(self, original: str, rewritten: str) -> List[str]:
        """Identify key changes made in rewriting."""
        changes = []
        
        # Check for added action verbs
        action_verbs = ["achieved", "developed", "implemented", "optimized", "increased"]
        for verb in action_verbs:
            if verb.lower() not in original.lower() and verb.lower() in rewritten.lower():
                changes.append(f"Added action verb: {verb}")
        
        # Check for added metrics
        original_numbers = re.findall(r'\d+', original)
        rewritten_numbers = re.findall(r'\d+', rewritten)
        
        if len(rewritten_numbers) > len(original_numbers):
            changes.append("Added quantifiable metrics")
        
        # Check for improved specificity
        if len(rewritten.split()) > len(original.split()) * 1.2:
            changes.append("Increased specificity and detail")
        
        return changes
    
    def load_model(self, model_path: str):
        """Load fine-tuned model."""
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model.eval()

def create_training_data_from_sources() -> pd.DataFrame:
    """
    Create training data from multiple sources:
    1. GPT-3.5 distilled pairs
    2. Reddit r/resumes improvements  
    3. ResumeWorded blog examples
    """
    training_pairs = []
    
    # Source 1: GPT distilled pairs (placeholder)
    gpt_pairs = [
        {
            "original": "Worked on web development projects",
            "improved": "Developed 5 responsive web applications using React and Node.js, improving user engagement by 30%",
            "jd_context": "Frontend Developer position requiring React experience"
        },
        {
            "original": "Managed team projects",
            "improved": "Led cross-functional team of 8 developers to deliver 3 major projects on time, reducing delivery timeline by 25%",
            "jd_context": "Team Lead position requiring project management skills"
        }
    ]
    
    # Source 2: Reddit improvements (placeholder)
    reddit_pairs = [
        {
            "original": "Used Python for data analysis",
            "improved": "Analyzed large datasets using Python and Pandas, identifying key trends that increased revenue by 15%",
            "jd_context": "Data Analyst position requiring Python and data analysis skills"
        }
    ]
    
    # Source 3: ResumeWorded examples (placeholder)
    resume_worded_pairs = [
        {
            "original": "Responsible for customer service",
            "improved": "Provided exceptional customer service to 50+ daily customers, achieving 95% satisfaction rating",
            "jd_context": "Customer Service Representative position"
        }
    ]
    
    # Combine all sources
    all_pairs = gpt_pairs + reddit_pairs + resume_worded_pairs
    
    # Expand with variations
    expanded_pairs = []
    for pair in all_pairs:
        # Add original pair
        expanded_pairs.append(pair)
        
        # Add variations with different JD contexts
        contexts = [
            "Software Engineer position",
            "Senior Developer role", 
            "Full Stack Developer position",
            "Data Scientist role"
        ]
        
        for context in contexts:
            if context != pair["jd_context"]:
                expanded_pairs.append({
                    "original": pair["original"],
                    "improved": pair["improved"],
                    "jd_context": context
                })
    
    return pd.DataFrame(expanded_pairs)

# Usage example
if __name__ == "__main__":
    # Create training data
    training_data = create_training_data_from_sources()
    print(f"Created {len(training_data)} training pairs")
    
    # Initialize rewriter
    rewriter = T5ResumeRewriter()
    
    # Train model (uncomment to actually train)
    # rewriter.train(training_data)
    
    # Test rewriting
    original_bullet = "Worked on machine learning projects"
    jd_context = "Data Scientist position requiring Python, TensorFlow, and ML experience"
    
    alternatives = rewriter.rewrite_bullet(original_bullet, jd_context, num_alternatives=3)
    
    print(f"Original: {original_bullet}")
    print(f"JD Context: {jd_context}")
    print("\nRewrite alternatives:")
    
    for i, alt in enumerate(alternatives, 1):
        print(f"{i}. {alt['rewritten']}")
        print(f"   Score: {alt['improvement_score']:.2f}")
        print(f"   Changes: {', '.join(alt['changes'])}")
        print()