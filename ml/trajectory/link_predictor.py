"""
Module 4a: Career Trajectory Predictor
GraphSAGE link prediction for career path recommendations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score

class CareerGraphSAGE(nn.Module):
    """
    GraphSAGE model for career trajectory prediction.
    Predicts next career moves based on current role and skills.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(CareerGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index):
        """Forward pass through GraphSAGE layers."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def predict_link(self, x, edge_index, src_nodes, dst_nodes):
        """Predict probability of links between source and destination nodes."""
        # Get node embeddings
        node_embeddings = self.forward(x, edge_index)
        
        # Get embeddings for source and destination nodes
        src_embeddings = node_embeddings[src_nodes]
        dst_embeddings = node_embeddings[dst_nodes]
        
        # Concatenate embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Predict link probability
        link_probs = self.link_predictor(edge_embeddings)
        
        return link_probs.squeeze()

class CareerTransitionGraph:
    """
    Builds career transition graph from career data.
    Nodes: job roles, edges: observed transitions with frequencies.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.role_to_id = {}
        self.id_to_role = {}
        self.transition_counts = {}
        
    def build_from_career_data(self, career_transitions: List[Tuple[str, str]]):
        """
        Build graph from career transition data.
        
        Args:
            career_transitions: List of (from_role, to_role) tuples
        """
        # Count transitions
        for from_role, to_role in career_transitions:
            if (from_role, to_role) not in self.transition_counts:
                self.transition_counts[(from_role, to_role)] = 0
            self.transition_counts[(from_role, to_role)] += 1
        
        # Create role mappings
        all_roles = set()
        for from_role, to_role in career_transitions:
            all_roles.add(from_role)
            all_roles.add(to_role)
        
        for i, role in enumerate(sorted(all_roles)):
            self.role_to_id[role] = i
            self.id_to_role[i] = role
        
        # Build graph
        for (from_role, to_role), count in self.transition_counts.items():
            from_id = self.role_to_id[from_role]
            to_id = self.role_to_id[to_role]
            
            # Add edge with transition probability as weight
            total_from_transitions = sum(
                c for (f, t), c in self.transition_counts.items() if f == from_role
            )
            transition_prob = count / total_from_transitions
            
            self.graph.add_edge(
                from_id, to_id,
                weight=transition_prob,
                count=count,
                from_role=from_role,
                to_role=to_role
            )
        
        # Add node attributes
        for role_id, role_name in self.id_to_role.items():
            self.graph.add_node(
                role_id,
                role=role_name,
                seniority=self._extract_seniority(role_name),
                domain=self._extract_domain(role_name)
            )
    
    def _extract_seniority(self, role_name: str) -> str:
        """Extract seniority level from role name."""
        role_lower = role_name.lower()
        
        if any(word in role_lower for word in ['senior', 'sr', 'lead', 'principal', 'staff']):
            return "senior"
        elif any(word in role_lower for word in ['junior', 'jr', 'entry', 'associate']):
            return "junior"
        else:
            return "mid"
    
    def _extract_domain(self, role_name: str) -> str:
        """Extract domain from role name."""
        role_lower = role_name.lower()
        
        if any(word in role_lower for word in ['data', 'scientist', 'analyst', 'ml', 'ai']):
            return "data_science"
        elif any(word in role_lower for word in ['devops', 'sre', 'infrastructure', 'cloud']):
            return "devops"
        elif any(word in role_lower for word in ['frontend', 'backend', 'fullstack', 'web']):
            return "web_development"
        elif any(word in role_lower for word in ['mobile', 'android', 'ios']):
            return "mobile"
        elif any(word in role_lower for word in ['product', 'manager', 'pm']):
            return "product"
        else:
            return "engineering"
    
    def get_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data format."""
        # Create node features
        node_features = []
        for node_id in range(len(self.id_to_role)):
            role_name = self.id_to_role[node_id]
            features = self._create_role_features(role_name)
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edge index and weights
        edge_list = list(self.graph.edges())
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor([
                self.graph[u][v]['weight'] for u, v in edge_list
            ], dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty((0,), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weights)
    
    def _create_role_features(self, role_name: str) -> List[float]:
        """Create feature vector for a role."""
        features = []
        
        # Seniority encoding
        seniority = self._extract_seniority(role_name)
        seniority_encoding = [0.0, 0.0, 0.0]  # [junior, mid, senior]
        if seniority == "junior":
            seniority_encoding[0] = 1.0
        elif seniority == "mid":
            seniority_encoding[1] = 1.0
        elif seniority == "senior":
            seniority_encoding[2] = 1.0
        features.extend(seniority_encoding)
        
        # Domain encoding
        domain = self._extract_domain(role_name)
        domains = ["data_science", "devops", "web_development", "mobile", "product", "engineering"]
        domain_encoding = [1.0 if domain == d else 0.0 for d in domains]
        features.extend(domain_encoding)
        
        # Role-specific features
        role_lower = role_name.lower()
        role_keywords = [
            "engineer", "developer", "scientist", "analyst", "manager",
            "lead", "architect", "consultant", "specialist"
        ]
        keyword_features = [1.0 if keyword in role_lower else 0.0 for keyword in role_keywords]
        features.extend(keyword_features)
        
        # Pad to fixed size
        target_size = 128
        if len(features) > target_size:
            features = features[:target_size]
        else:
            features.extend([0.0] * (target_size - len(features)))
        
        return features

class CareerTrajectoryPredictor:
    """
    Complete career trajectory prediction system.
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.career_graph = CareerTransitionGraph()
        
        if model_path:
            self.load_model(model_path)
    
    def train(self, career_transitions: List[Tuple[str, str]], 
              save_path: str = "ml/models/career_trajectory.pt"):
        """
        Train the career trajectory model.
        
        Args:
            career_transitions: List of (from_role, to_role) transitions
        """
        # Build career graph
        self.career_graph.build_from_career_data(career_transitions)
        data = self.career_graph.get_pyg_data()
        
        # Initialize model
        self.model = CareerGraphSAGE(
            input_dim=data.x.size(1),
            hidden_dim=64,
            output_dim=32,
            num_layers=2
        )
        
        # Prepare training data
        train_edges, train_labels = self._prepare_training_data(data)
        
        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            
            # Predict link probabilities
            src_nodes, dst_nodes = train_edges
            link_probs = self.model.predict_link(data.x, data.edge_index, src_nodes, dst_nodes)
            
            # Compute loss
            loss = criterion(link_probs, train_labels.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'career_graph_data': nx.node_link_data(self.career_graph.graph),
            'model_config': {
                'input_dim': data.x.size(1),
                'hidden_dim': 64,
                'output_dim': 32,
                'num_layers': 2
            }
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def predict_next_roles(self, current_role: str, top_k: int = 3) -> List[Dict]:
        """
        Predict top-k next career moves for a given role.
        
        Returns:
            List of predicted roles with probabilities and metadata
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if current_role not in self.career_graph.role_to_id:
            return []  # Role not in training data
        
        current_role_id = self.career_graph.role_to_id[current_role]
        data = self.career_graph.get_pyg_data()
        
        # Get all possible destination roles
        all_role_ids = list(self.career_graph.id_to_role.keys())
        src_nodes = torch.tensor([current_role_id] * len(all_role_ids))
        dst_nodes = torch.tensor(all_role_ids)
        
        # Predict transition probabilities
        self.model.eval()
        with torch.no_grad():
            transition_probs = self.model.predict_link(data.x, data.edge_index, src_nodes, dst_nodes)
        
        # Get top-k predictions
        top_k_indices = torch.topk(transition_probs, min(top_k, len(all_role_ids))).indices
        
        predictions = []
        for idx in top_k_indices:
            role_id = all_role_ids[idx]
            role_name = self.career_graph.id_to_role[role_id]
            probability = transition_probs[idx].item()
            
            predictions.append({
                "role": role_name,
                "probability": probability,
                "seniority": self.career_graph._extract_seniority(role_name),
                "domain": self.career_graph._extract_domain(role_name),
                "transition_time_estimate": self._estimate_transition_time(current_role, role_name)
            })
        
        return predictions
    
    def _prepare_training_data(self, data: Data) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Prepare positive and negative edges for training."""
        # Positive edges (existing transitions)
        pos_edges = data.edge_index
        pos_labels = torch.ones(pos_edges.size(1))
        
        # Negative edges (random non-existing edges)
        num_nodes = data.x.size(0)
        neg_edges = []
        existing_edges = set(map(tuple, pos_edges.t().tolist()))
        
        for _ in range(pos_edges.size(1)):
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            
            # Ensure it's not an existing edge
            if src != dst and (src, dst) not in existing_edges and [src, dst] not in neg_edges:
                neg_edges.append([src, dst])
        
        if neg_edges:
            neg_edges = torch.tensor(neg_edges).t()
            neg_labels = torch.zeros(neg_edges.size(1))
        else:
            neg_edges = torch.empty((2, 0), dtype=torch.long)
            neg_labels = torch.empty((0,))
        
        # Combine positive and negative edges
        all_edges = torch.cat([pos_edges, neg_edges], dim=1)
        all_labels = torch.cat([pos_labels, neg_labels])
        
        return (all_edges[0], all_edges[1]), all_labels
    
    def _estimate_transition_time(self, from_role: str, to_role: str) -> str:
        """Estimate time required for career transition."""
        from_seniority = self.career_graph._extract_seniority(from_role)
        to_seniority = self.career_graph._extract_seniority(to_role)
        from_domain = self.career_graph._extract_domain(from_role)
        to_domain = self.career_graph._extract_domain(to_role)
        
        # Simple heuristic for transition time
        if from_domain != to_domain:
            return "2-3 years"  # Domain change takes longer
        elif from_seniority == "junior" and to_seniority == "senior":
            return "3-5 years"  # Seniority jump
        elif from_seniority == to_seniority:
            return "1-2 years"  # Lateral move
        else:
            return "2-3 years"  # Default
    
    def load_model(self, model_path: str):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        self.career_graph = CareerTransitionGraph()
        self.career_graph.graph = nx.node_link_graph(checkpoint['career_graph_data'])
        self.career_graph.role_to_id = {attrs['role']: node for node, attrs in self.career_graph.graph.nodes(data=True)}
        self.career_graph.id_to_role = {node: role for role, node in self.career_graph.role_to_id.items()}
        model_config = checkpoint['model_config']
        
        self.model = CareerGraphSAGE(**model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

# Usage example
if __name__ == "__main__":
    # Sample career transition data
    career_transitions = [
        ("Junior Software Engineer", "Software Engineer"),
        ("Software Engineer", "Senior Software Engineer"),
        ("Senior Software Engineer", "Tech Lead"),
        ("Tech Lead", "Engineering Manager"),
        ("Data Analyst", "Data Scientist"),
        ("Data Scientist", "Senior Data Scientist"),
        ("Senior Data Scientist", "ML Engineer"),
        ("DevOps Engineer", "Senior DevOps Engineer"),
        ("Senior DevOps Engineer", "Platform Engineer"),
        ("Frontend Developer", "Full Stack Developer"),
        ("Full Stack Developer", "Senior Full Stack Developer"),
        ("Software Engineer", "Product Manager"),  # Cross-domain transition
        ("Data Scientist", "Product Manager"),     # Cross-domain transition
    ] * 50  # Repeat for more data
    
    # Initialize predictor
    predictor = CareerTrajectoryPredictor()
    
    # Train model
    predictor.train(career_transitions)
    
    # Make predictions
    current_role = "Software Engineer"
    predictions = predictor.predict_next_roles(current_role, top_k=3)
    
    print(f"Career predictions for '{current_role}':")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['role']} (probability: {pred['probability']:.3f}, "
              f"time: {pred['transition_time_estimate']})")
