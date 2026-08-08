"""
T5 fine-tuning with GPT distillation for resume rewriting.
Knowledge distillation from teacher (GPT) to student (T5).
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
import torch

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
        
        # Format as T5 task
        input_text = f"rewrite resume bullet: {row['original']}"
        target_text = row['improved']
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
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

def train_t5_rewriter():
    """
    Fine-tune T5 on GPT-distilled resume improvements.
    This is legitimate knowledge distillation, not API dependency.
    """
    
    # Load distilled training data
    pairs_df = pd.read_csv('data/t5_training_data.csv')
    print(f"Loaded {len(pairs_df)} training pairs from GPT distillation")
    
    # Initialize T5
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Create dataset
    train_dataset = ResumeRewriteDataset(pairs_df, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./t5_resume_rewriter',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting T5 fine-tuning with distilled data...")
    trainer.train()
    
    # Save
    model.save_pretrained('./t5_resume_rewriter_final')
    tokenizer.save_pretrained('./t5_resume_rewriter_final')
    
    print("T5 rewriter training complete!")

if __name__ == "__main__":
    train_t5_rewriter()