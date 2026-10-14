"""
Graph Converter - NetworkX to PyTorch Geometric conversion
"""

import torch
import networkx as nx
from torch_geometric.data import HeteroData, Data
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class GraphConverter:
    """
    Converts NetworkX graphs to PyTorch Geometric format for GAT training.
    Handles heterogeneous graphs with different node types.
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        # Explicit lexical features; do not download BERT and never use it.
        
        # Node type mappings
        self.node_types = ["SKILL", "ROLE", "TOOL", "EDUCATION", "COMPANY"]
        self.edge_types = ["RELATED_TO", "REQUIRES", "HAS_SKILL", "WORKED_AT"]
    
    def networkx_to_pyg(self, G: nx.DiGraph) -> HeteroData:
        """
        Convert NetworkX graph to PyTorch Geometric HeteroData.
        
        Args:
            G: NetworkX directed graph
            
        Returns:
            PyTorch Geometric HeteroData object
        """
        data = HeteroData()
        
        # Group nodes by type
        nodes_by_type = self._group_nodes_by_type(G)
        
        # Add node features for each type
        for node_type, nodes in nodes_by_type.items():
            if nodes:
                node_features = self._create_node_features(G, nodes, node_type)
                data[node_type].x = torch.tensor(node_features, dtype=torch.float)
                data[node_type].num_nodes = len(nodes)
        
        # Add edges
        self._add_edges_to_hetero_data(G, data, nodes_by_type)
        
        return data
    
    def create_homogeneous_graph(self, G: nx.DiGraph) -> Data:
        """
        Convert to homogeneous PyTorch Geometric Data for simpler GAT models.
        All nodes get the same feature representation.
        """
        # Create node mapping
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Create node features
        node_features = []
        for node in node_list:
            features = self._create_single_node_features(G, node)
            node_features.append(features)

        if not node_features:
            raise ValueError("No recognized entities in graph; cannot encode an empty graph.")
        
        x = torch.from_numpy(np.asarray(node_features, dtype=np.float32))
        
        # Create edge index
        edge_index = []
        edge_attr = []
        
        for source, target, edge_data in G.edges(data=True):
            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]
            
            edge_index.append([source_idx, target_idx])
            edge_attr.append(edge_data.get('weight', 1.0))
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _group_nodes_by_type(self, G: nx.DiGraph) -> Dict[str, List[str]]:
        """Group nodes by their type attribute."""
        nodes_by_type = {node_type: [] for node_type in self.node_types}
        
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'UNKNOWN')
            if node_type in nodes_by_type:
                nodes_by_type[node_type].append(node)
        
        return nodes_by_type
    
    def _create_node_features(self, G: nx.DiGraph, nodes: List[str], node_type: str) -> np.ndarray:
        """Create feature matrix for nodes of a specific type."""
        features = []
        
        for node in nodes:
            node_data = G.nodes[node]
            feature_vector = self._create_single_node_features(G, node)
            features.append(feature_vector)
        
        return np.array(features)
    
    def _create_single_node_features(self, G: nx.DiGraph, node: str) -> np.ndarray:
        """Create feature vector for a single node."""
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'UNKNOWN')
        
        # Base features
        features = []
        
        # One-hot encoding for node type
        type_encoding = [0.0] * len(self.node_types)
        if node_type in self.node_types:
            type_encoding[self.node_types.index(node_type)] = 1.0
        features.extend(type_encoding)
        
        # Node-specific features
        features.extend([
            node_data.get('confidence', 0.5),
            node_data.get('importance', 0.5),
            node_data.get('weight', 1.0),
            G.degree(node),  # Node degree
            G.in_degree(node),  # In-degree
            G.out_degree(node),  # Out-degree
        ])
        
        # Text embedding (simplified - using name/title)
        text = node_data.get('name', node_data.get('title', node))
        text_embedding = self._get_text_embedding(text)
        features.extend(text_embedding)
        
        # Pad or truncate to fixed size
        target_size = self.embedding_dim
        if len(features) > target_size:
            features = features[:target_size]
        else:
            features.extend([0.0] * (target_size - len(features)))
        
        return np.array(features)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get BERT embedding for text (simplified version)."""
        # For production, use full BERT embeddings
        # Here we use a simple hash-based embedding for speed
        
        # Simple TF-IDF style features
        words = text.lower().split()
        
        # Common career keywords
        career_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue',
            'aws', 'docker', 'kubernetes', 'sql', 'nosql', 'machine learning',
            'data science', 'backend', 'frontend', 'fullstack', 'devops',
            'senior', 'junior', 'lead', 'manager', 'engineer', 'developer'
        ]
        
        # Binary features for keyword presence
        keyword_features = [1.0 if keyword in text.lower() else 0.0 for keyword in career_keywords]
        
        # Text length features
        length_features = [
            len(text),
            len(words),
            len(set(words)),  # Unique words
        ]
        
        # Combine features
        embedding = keyword_features + length_features
        
        # Pad to reasonable size
        target_size = 50
        if len(embedding) > target_size:
            embedding = embedding[:target_size]
        else:
            embedding.extend([0.0] * (target_size - len(embedding)))
        
        return embedding
    
    def _add_edges_to_hetero_data(self, G: nx.DiGraph, data: HeteroData, nodes_by_type: Dict):
        """Add edges to heterogeneous data object."""
        # Create node index mappings for each type
        node_mappings = {}
        for node_type, nodes in nodes_by_type.items():
            node_mappings[node_type] = {node: idx for idx, node in enumerate(nodes)}
        
        # Group edges by type
        edges_by_type = {}
        
        for source, target, edge_data in G.edges(data=True):
            source_type = G.nodes[source]['type']
            target_type = G.nodes[target]['type']
            if source_type not in node_mappings or target_type not in node_mappings:
                continue
            edge_type = edge_data.get('type', 'RELATED_TO')
            
            edge_key = (source_type, edge_type, target_type)
            
            if edge_key not in edges_by_type:
                edges_by_type[edge_key] = []
            
            source_idx = node_mappings[source_type][source]
            target_idx = node_mappings[target_type][target]
            weight = edge_data.get('weight', 1.0)
            
            edges_by_type[edge_key].append([source_idx, target_idx, weight])
        
        # Add edges to data object
        for (source_type, edge_type, target_type), edges in edges_by_type.items():
            if edges:
                edge_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t()
                edge_attr = torch.tensor([e[2] for e in edges], dtype=torch.float)
                
                data[source_type, edge_type, target_type].edge_index = edge_index
                data[source_type, edge_type, target_type].edge_attr = edge_attr

# Usage example
if __name__ == "__main__":
    from ml.graph.builder import KnowledgeGraphBuilder
    
    # Create sample graph
    builder = KnowledgeGraphBuilder()
    text = "Senior Python Developer with React and AWS experience"
    G = builder.build_graph(text)
    
    # Convert to PyTorch Geometric
    converter = GraphConverter()
    
    # Homogeneous version (simpler)
    homo_data = converter.create_homogeneous_graph(G)
    print(f"Homogeneous graph: {homo_data.x.shape[0]} nodes, {homo_data.edge_index.shape[1]} edges")
    
    # Heterogeneous version (more complex but potentially better)
    hetero_data = converter.networkx_to_pyg(G)
    print(f"Heterogeneous graph node types: {list(hetero_data.node_types)}")
    print(f"Heterogeneous graph edge types: {list(hetero_data.edge_types)}")
