"""
GAT Training Script with Contrastive Learning
Trains the flagship Graph Attention Network on synthetic hard negatives.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
try:
    import wandb
except ImportError:
    wandb = None
from tqdm import tqdm
from typing import Dict, Tuple

from ml.gat.model import SiameseGAT, ContrastiveLoss
from ml.gat.dataset import ContrastiveResumeDataset, collate_graph_batch, create_weak_supervision_pairs

class GATTrainer:
    """
    Trainer for Graph Attention Network with contrastive learning.
    Implements InfoNCE loss with synthetic hard negatives.
    """
    
    def __init__(
        self,
        model_config: Dict,
        training_config: Dict,
        use_wandb: bool = True
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.use_wandb = use_wandb
        
        # Initialize model
        self.model = SiameseGAT(model_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        self.criterion = ContrastiveLoss(temperature=training_config.get('temperature', 0.07))
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 1e-2)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # Initialize wandb
        if self.use_wandb and wandb is not None:
            wandb.init(
                project="careerlens-gat",
                config={**model_config, **training_config}
            )
    
    def train(
        self,
        train_dataset: ContrastiveResumeDataset,
        val_dataset: ContrastiveResumeDataset,
        save_path: str = "ml/models/gat_model.pt"
    ):
        """
        Train the GAT model with contrastive learning.
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            collate_fn=collate_graph_batch,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            collate_fn=collate_graph_batch,
            num_workers=0
        )
        
        best_val_recall = 0.0
        patience_counter = 0
        
        for epoch in range(self.training_config['num_epochs']):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['recall_at_5'])
            
            # Logging
            print(f"Epoch {epoch+1}/{self.training_config['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Recall@5: {val_metrics['recall_at_5']:.4f}")
            
            if self.use_wandb and wandb is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_metrics['accuracy'],
                    "val_recall_at_5": val_metrics['recall_at_5'],
                    "val_mrr": val_metrics['mrr'],
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Early stopping and model saving
            if val_metrics['recall_at_5'] > best_val_recall:
                best_val_recall = val_metrics['recall_at_5']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'model_config': self.model_config,
                    'epoch': epoch,
                    'best_val_recall': best_val_recall
                }, save_path)
                
                print(f"New best model saved with Recall@5: {best_val_recall:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.training_config.get('patience', 10):
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
        
        print(f"Training completed. Best Recall@5: {best_val_recall:.4f}")
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_similarities = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (resume_batch, jd_batch, labels) in enumerate(pbar):
            # Move to device
            resume_batch = resume_batch.to(self.device)
            jd_batch = jd_batch.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            resume_emb, jd_emb, similarities = self.model(resume_batch, jd_batch)
            
            # Compute loss
            loss = self.criterion(resume_emb, jd_emb, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            all_similarities.extend(similarities.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(all_similarities, all_labels)
        
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_similarities = []
        all_labels = []
        
        with torch.no_grad():
            for resume_batch, jd_batch, labels in tqdm(val_loader, desc="Validation"):
                # Move to device
                resume_batch = resume_batch.to(self.device)
                jd_batch = jd_batch.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                resume_emb, jd_emb, similarities = self.model(resume_batch, jd_batch)
                
                # Compute loss
                loss = self.criterion(resume_emb, jd_emb, labels)
                
                # Accumulate metrics
                total_loss += loss.item()
                all_similarities.extend(similarities.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(all_similarities, all_labels)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, similarities: list, labels: list) -> Dict:
        """Calculate evaluation metrics."""
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        # Binary classification metrics
        predictions = (similarities > 0.5).astype(int)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        # Ranking metrics
        recall_at_5 = self._calculate_recall_at_k(similarities, labels, k=5)
        mrr = self._calculate_mrr(similarities, labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'recall_at_5': recall_at_5,
            'mrr': mrr
        }
    
    def _calculate_recall_at_k(self, similarities: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
        """Calculate Recall@K metric."""
        # Sort by similarity scores (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Get top-k predictions
        top_k_labels = labels[sorted_indices[:k]]
        
        # Calculate recall@k
        if np.sum(labels) == 0:  # No positive examples
            return 0.0
        
        recall_at_k = np.sum(top_k_labels) / min(k, np.sum(labels))
        return recall_at_k
    
    def _calculate_mrr(self, similarities: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Mean Reciprocal Rank."""
        # Sort by similarity scores (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_labels = labels[sorted_indices]
        
        # Find first positive example
        first_positive = np.where(sorted_labels == 1)[0]
        
        if len(first_positive) == 0:
            return 0.0
        
        # MRR is 1 / (rank of first positive)
        mrr = 1.0 / (first_positive[0] + 1)
        return mrr

def load_training_data() -> Tuple[ContrastiveResumeDataset, ContrastiveResumeDataset]:
    """
    Load and prepare training data from Kaggle datasets.
    """
    # Load datasets (placeholder - replace with actual data loading)
    print("Loading Kaggle datasets...")
    
    # In practice, load from:
    # resumes_df = pd.read_csv('data/kaggle_resumes.csv')
    # jds_df = pd.read_csv('data/linkedin_jobs.csv')
    
    # For demo, create sample data
    sample_resumes = [
        "Senior Python Developer with 5 years Django and React experience",
        "Data Scientist with machine learning and TensorFlow expertise",
        "DevOps Engineer with AWS, Docker, and Kubernetes experience",
        "Full Stack Developer with JavaScript, Node.js, and MongoDB skills",
        "Machine Learning Engineer with PyTorch and computer vision experience"
    ] * 100  # Repeat for larger dataset
    
    sample_jds = [
        "Python Developer position requiring Django and web development skills",
        "Data Scientist role with ML and Python requirements",
        "DevOps position with cloud and containerization experience",
        "Full Stack role requiring JavaScript and database experience",
        "ML Engineer position with deep learning and Python skills"
    ] * 100
    
    # Create weak supervision labels
    labels = [1, 1, 1, 1, 1] * 100  # All positive for demo
    
    # Split into train/val
    split_idx = int(0.8 * len(sample_resumes))
    
    train_dataset = ContrastiveResumeDataset(
        resume_texts=sample_resumes[:split_idx],
        jd_texts=sample_jds[:split_idx],
        labels=labels[:split_idx],
        hard_negative_ratio=3.0,
        skill_removal_count=3
    )
    
    val_dataset = ContrastiveResumeDataset(
        resume_texts=sample_resumes[split_idx:],
        jd_texts=sample_jds[split_idx:],
        labels=labels[split_idx:],
        hard_negative_ratio=2.0,
        skill_removal_count=2
    )
    
    return train_dataset, val_dataset

def main():
    """Main training function."""
    # Model configuration
    model_config = {
        "input_dim": 768,
        "hidden_dim": 128,
        "output_dim": 64,
        "num_heads": 8,
        "num_layers": 3,
        "dropout": 0.3
    }
    
    # Training configuration
    training_config = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 50,
        "weight_decay": 1e-2,
        "temperature": 0.07,
        "patience": 10
    }
    
    # Load data
    train_dataset, val_dataset = load_training_data()
    
    # Initialize trainer
    trainer = GATTrainer(model_config, training_config, use_wandb=False)
    
    # Create models directory
    os.makedirs("ml/models", exist_ok=True)
    
    # Train model
    trainer.train(train_dataset, val_dataset, "ml/models/gat_contrastive.pt")

if __name__ == "__main__":
    main()