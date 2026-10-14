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
    Uses labeled contrastive pairs with explicit binary loss.
    """
    
    def __init__(
        self,
        model_config: Dict,
        training_config: Dict,
        use_wandb: bool = False
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
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        if not len(train_dataset) or not len(val_dataset):
            raise ValueError('Training and validation datasets cannot be empty.')
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
        
        best_val_f1 = -1.0
        patience_counter = 0
        
        for epoch in range(self.training_config['num_epochs']):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['f1'])
            
            # Logging
            print(f"Epoch {epoch+1}/{self.training_config['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            if self.use_wandb and wandb is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_metrics['accuracy'],
                    "val_f1": val_metrics['f1'],
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Early stopping and model saving
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'model_config': self.model_config,
                    'epoch': epoch,
                    'best_val_f1': best_val_f1
                }, save_path)
                
                print(f"New best model saved with F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.training_config.get('patience', 10):
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
        
        print(f"Training completed. Best F1: {best_val_f1:.4f}")
    
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
            labels, predictions, average='binary', zero_division=0
        )
        
        # Pair classification, not a per-job ranking benchmark.
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    

def load_training_data() -> Tuple[ContrastiveResumeDataset, ContrastiveResumeDataset]:
    """
    Load and prepare training data from Kaggle datasets.
    """
    from sklearn.model_selection import GroupShuffleSplit
    from ml.skills import extract_skills
    path = os.getenv("GAT_DATASET", "data/gat_training_data.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Provide a labeled CSV at {path} with resume,jd,label columns. No toy data is used.")
    frame = pd.read_csv(path)
    if not {"resume", "jd", "label"}.issubset(frame.columns):
        raise ValueError("Dataset needs resume, jd, label columns.")
    if frame.groupby(["resume", "jd"])["label"].nunique().gt(1).any():
        raise ValueError("The same pair has conflicting labels; review the source dataset.")
    frame = frame.drop_duplicates(subset=["resume", "jd"]).reset_index(drop=True)
    if frame[["resume", "jd", "label"]].isna().any().any() or not frame["label"].isin([0, 1]).all():
        raise ValueError("Dataset must have nonempty text and binary labels.")
    frame = frame[frame["resume"].map(lambda s: bool(extract_skills(str(s)))) &
                  frame["jd"].map(lambda s: bool(extract_skills(str(s))))].reset_index(drop=True)
    if frame["resume"].nunique() < 5:
        raise ValueError("At least five distinct resumes with recognized skills are needed for a grouped split.")
    # Hold out entire resumes; synthetic variants are produced only after splitting.
    for train_index, val_index in GroupShuffleSplit(n_splits=30, test_size=0.2, random_state=42).split(frame, groups=frame["resume"]):
        train, val = frame.iloc[train_index], frame.iloc[val_index]
        if train["label"].nunique() == 2 and val["label"].nunique() == 2:
            break
    else:
        raise ValueError("Both train and validation sets need positive and negative labels.")
    def dataset(rows, ratio):
        return ContrastiveResumeDataset(rows["resume"].tolist(), rows["jd"].tolist(), rows["label"].tolist(),
                                        hard_negative_ratio=ratio)
    return dataset(train, 1.0), dataset(val, 0.0)


def main():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
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
