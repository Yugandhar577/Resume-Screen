"""
Module 2: GNN Job Matching Engine (FLAGSHIP)
Graph Attention Network for semantic resume-JD matching with contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional
import numpy as np

class ResumeGAT(nn.Module):
    """
    Graph Attention Network for resume-JD matching.
    Two-tower Siamese architecture with shared weights.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super(ResumeGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(
                input_dim, 
                hidden_dim, 
                heads=num_heads, 
                dropout=dropout,
                concat=True
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
        # Final layer
        self.gat_layers.append(
            GATConv(
                hidden_dim * num_heads,
                output_dim,
                heads=1,
                dropout=dropout,
                concat=False
            )
        )
        
        # Graph-level pooling
        self.pool_type = "attention"  # "mean", "max", or "attention"
        
        if self.pool_type == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=0)
            )
        
        # Final projection layer
        self.projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through GAT layers.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for each node [num_nodes]
            
        Returns:
            Graph-level embeddings [batch_size, output_dim]
        """
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            
            if i < len(self.gat_layers) - 1:  # Not the last layer
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        if batch is None:
            # Single graph
            if self.pool_type == "mean":
                graph_emb = torch.mean(x, dim=0, keepdim=True)
            elif self.pool_type == "max":
                graph_emb = torch.max(x, dim=0, keepdim=True)[0]
            elif self.pool_type == "attention":
                attention_weights = self.attention_pool(x)
                graph_emb = torch.sum(attention_weights * x, dim=0, keepdim=True)
        else:
            # Batch of graphs
            if self.pool_type == "mean":
                graph_emb = global_mean_pool(x, batch)
            elif self.pool_type == "max":
                graph_emb = global_max_pool(x, batch)
            elif self.pool_type == "attention":
                # Custom attention pooling for batches
                graph_emb = self._attention_pool_batch(x, batch)
        
        # Final projection and normalization
        graph_emb = self.projection(graph_emb)
        graph_emb = F.normalize(graph_emb, p=2, dim=1)
        
        return graph_emb
    
    def _attention_pool_batch(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Custom attention pooling for batched graphs."""
        batch_size = batch.max().item() + 1
        graph_embeddings = []
        
        for i in range(batch_size):
            mask = batch == i
            graph_nodes = x[mask]
            
            if graph_nodes.size(0) > 0:
                attention_weights = self.attention_pool(graph_nodes)
                graph_emb = torch.sum(attention_weights * graph_nodes, dim=0)
                graph_embeddings.append(graph_emb)
            else:
                # Empty graph
                graph_embeddings.append(torch.zeros(self.output_dim, device=x.device))
        
        return torch.stack(graph_embeddings)

class SiameseGAT(nn.Module):
    """
    Siamese GAT for resume-JD matching.
    Shared encoder for both resume and JD graphs.
    """
    
    def __init__(self, gat_config: dict):
        super(SiameseGAT, self).__init__()
        
        # Shared GAT encoder
        self.encoder = ResumeGAT(**gat_config)
        
        # Similarity computation
        self.similarity_type = "cosine"  # "cosine", "dot", or "mlp"
        
        if self.similarity_type == "mlp":
            self.similarity_mlp = nn.Sequential(
                nn.Linear(gat_config["output_dim"] * 2, gat_config["hidden_dim"]),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(gat_config["hidden_dim"], 1),
                nn.Sigmoid()
            )
    
    def forward(self, resume_data: Data, jd_data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for resume-JD pair.
        
        Args:
            resume_data: PyTorch Geometric Data for resume
            jd_data: PyTorch Geometric Data for JD
            
        Returns:
            resume_emb: Resume graph embedding
            jd_emb: JD graph embedding  
            similarity: Similarity score
        """
        # Encode both graphs
        resume_emb = self.encoder(resume_data.x, resume_data.edge_index, resume_data.batch)
        jd_emb = self.encoder(jd_data.x, jd_data.edge_index, jd_data.batch)
        
        # Compute similarity
        if self.similarity_type == "cosine":
            similarity = F.cosine_similarity(resume_emb, jd_emb, dim=1)
        elif self.similarity_type == "dot":
            similarity = torch.sum(resume_emb * jd_emb, dim=1)
        elif self.similarity_type == "mlp":
            combined = torch.cat([resume_emb, jd_emb], dim=1)
            similarity = self.similarity_mlp(combined).squeeze()
        
        return resume_emb, jd_emb, similarity
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Get graph embedding for a single graph."""
        return self.encoder(data.x, data.edge_index, data.batch)

class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for resume-JD matching.
    Pulls together positive pairs, pushes apart hard negatives.
    """
    
    def __init__(self, temperature: float = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, resume_emb: torch.Tensor, jd_emb: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            resume_emb: Resume embeddings [batch_size, emb_dim]
            jd_emb: JD embeddings [batch_size, emb_dim]
            labels: Binary labels [batch_size] (1 for positive, 0 for negative)
            
        Returns:
            Contrastive loss
        """
        batch_size = resume_emb.size(0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(resume_emb, jd_emb.T) / self.temperature
        
        # Create positive pair mask
        positive_mask = labels.unsqueeze(1) * labels.unsqueeze(0)
        
        # Create targets for positive pairs
        targets = torch.arange(batch_size, device=resume_emb.device)
        
        # Mask out negative pairs in similarity matrix
        masked_similarity = similarity_matrix * positive_mask + (1 - positive_mask) * (-1e9)
        
        # Cross-entropy loss
        loss = F.cross_entropy(masked_similarity, targets)
        
        return loss

class GATExplainer:
    """
    Explainability for GAT model using attention weights.
    Shows which nodes/skills drove the matching decision.
    """
    
    def __init__(self, model: SiameseGAT):
        self.model = model
        self.model.eval()
    
    def explain_match(self, resume_data: Data, jd_data: Data) -> dict:
        """
        Explain why a resume matches (or doesn't match) a JD.
        
        Returns:
            Dictionary with attention weights and important nodes
        """
        with torch.no_grad():
            # Get embeddings and attention weights
            resume_emb, jd_emb, similarity = self.model(resume_data, jd_data)
            
            # Extract attention weights from GAT layers
            attention_weights = self._extract_attention_weights(resume_data, jd_data)
            
            # Find most important nodes
            important_resume_nodes = self._get_important_nodes(
                resume_data, attention_weights["resume"]
            )
            important_jd_nodes = self._get_important_nodes(
                jd_data, attention_weights["jd"]
            )
            
            return {
                "similarity_score": similarity.item(),
                "resume_embedding": resume_emb.cpu().numpy(),
                "jd_embedding": jd_emb.cpu().numpy(),
                "important_resume_nodes": important_resume_nodes,
                "important_jd_nodes": important_jd_nodes,
                "attention_weights": attention_weights
            }
    
    def _extract_attention_weights(self, resume_data: Data, jd_data: Data) -> dict:
        """Extract attention weights from GAT layers."""
        # This is a simplified version - in practice, you'd need to modify
        # the GAT forward pass to return attention weights
        
        # Placeholder implementation
        return {
            "resume": torch.rand(resume_data.x.size(0)),
            "jd": torch.rand(jd_data.x.size(0))
        }
    
    def _get_important_nodes(self, data: Data, attention_weights: torch.Tensor) -> list:
        """Get top-k most important nodes based on attention."""
        k = min(5, data.x.size(0))  # Top 5 or all nodes if fewer
        
        top_indices = torch.topk(attention_weights, k).indices
        
        important_nodes = []
        for idx in top_indices:
            # In practice, you'd have node metadata to extract names
            important_nodes.append({
                "index": idx.item(),
                "attention_weight": attention_weights[idx].item(),
                "features": data.x[idx].cpu().numpy()
            })
        
        return important_nodes

# Usage example
if __name__ == "__main__":
    # Model configuration
    gat_config = {
        "input_dim": 768,
        "hidden_dim": 128,
        "output_dim": 64,
        "num_heads": 8,
        "num_layers": 3,
        "dropout": 0.3
    }
    
    # Create model
    model = SiameseGAT(gat_config)
    
    # Create dummy data
    resume_data = Data(
        x=torch.randn(10, 768),
        edge_index=torch.randint(0, 10, (2, 15))
    )
    
    jd_data = Data(
        x=torch.randn(8, 768),
        edge_index=torch.randint(0, 8, (2, 12))
    )
    
    # Forward pass
    resume_emb, jd_emb, similarity = model(resume_data, jd_data)
    
    print(f"Resume embedding shape: {resume_emb.shape}")
    print(f"JD embedding shape: {jd_emb.shape}")
    print(f"Similarity score: {similarity.item():.4f}")
    
    # Test explainer
    explainer = GATExplainer(model)
    explanation = explainer.explain_match(resume_data, jd_data)
    print(f"Explanation keys: {list(explanation.keys())}")