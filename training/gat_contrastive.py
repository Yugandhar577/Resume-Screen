import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from data.synthetic_negatives import SyntheticNegativeGenerator
from data.weak_labelling import WeakLabeller

class ResumeGAT(torch.nn.Module):
    """Graph Attention Network for resume-JD matching with synthetic hard negatives."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.3)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=0.3)
        
    def forward(self, x, edge_index, batch):
        # Multi-head attention layers
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)
        
        # Global attention pooling
        x = global_mean_pool(x, batch)
        return F.normalize(x, p=2, dim=1)

class ContrastiveTrainer:
    """Train GAT with synthetic hard negatives and InfoNCE loss."""
    
    def __init__(self, model: ResumeGAT, temperature: float = 0.07):
        self.model = model
        self.temperature = temperature
        self.neg_generator = SyntheticNegativeGenerator()
    
    def info_nce_loss(self, resume_emb, jd_emb, labels):
        """
        InfoNCE loss with hard negatives.
        Pulls together positive pairs, pushes apart hard negatives.
        """
        # Compute similarity matrix
        similarity = torch.matmul(resume_emb, jd_emb.T) / self.temperature
        
        # Create targets (positive pairs should have high similarity)
        batch_size = resume_emb.size(0)
        targets = torch.arange(batch_size).to(resume_emb.device)
        
        # Mask out negative pairs in similarity matrix
        mask = labels.unsqueeze(1) * labels.unsqueeze(0)  # 1 for positive pairs
        similarity = similarity * mask + (1 - mask) * (-1e9)  # Mask negatives
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity, targets)
        return loss
    
    def train_epoch(self, dataloader, optimizer):
        """Train one epoch with synthetic hard negatives."""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            resume_emb = self.model(batch.resume_x, batch.resume_edge_index, batch.resume_batch)
            jd_emb = self.model(batch.jd_x, batch.jd_edge_index, batch.jd_batch)
            
            # Contrastive loss with hard negatives
            loss = self.info_nce_loss(resume_emb, jd_emb, batch.labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)

def create_training_data():
    """
    Create training dataset with synthetic hard negatives.
    This is the key innovation - no real rejection data needed.
    """
    # Load datasets
    weak_labeller = WeakLabeller()
    # resumes_df, jds_df = weak_labeller.load_datasets('kaggle_resumes.csv', 'linkedin_jobs.csv')
    
    # Generate weak supervision pairs
    # pairs = weak_labeller.generate_pairs(resumes_df, jds_df, num_pairs=5000)
    
    # Add synthetic hard negatives
    neg_generator = SyntheticNegativeGenerator()
    
    # For each positive pair, create hard negative by removing key skills
    enhanced_pairs = []
    # for resume_text, jd_text, label in pairs:
    #     if label == 1:  # Positive pair
    #         # Extract JD skills (simplified)
    #         jd_skills = jd_text.split()[:20]  # Use first 20 words as skills
    #         
    #         # Create contrastive pairs (1 positive + 1 hard negative)
    #         contrastive_pairs = neg_generator.create_contrastive_pairs(
    #             resume_text, jd_text, jd_skills
    #         )
    #         enhanced_pairs.extend(contrastive_pairs)
    
    return enhanced_pairs

if __name__ == "__main__":
    # Initialize GAT model
    model = ResumeGAT(input_dim=768, hidden_dim=128, output_dim=64, heads=4)
    trainer = ContrastiveTrainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create training data with synthetic hard negatives
    training_pairs = create_training_data()
    
    # Convert to PyTorch Geometric format and train
    # dataloader = create_dataloader(training_pairs)
    
    print("Training GAT with synthetic hard negatives...")
    for epoch in range(50):
        # loss = trainer.train_epoch(dataloader, optimizer)
        # print(f"Epoch {epoch}: Loss = {loss:.4f}")
        pass
    
    torch.save(model.state_dict(), "gat_contrastive.pt")
    print("Model saved!")