"""
Knowledge Graph Builder - Converts extracted entities into NetworkX graphs
"""

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False

from typing import Dict, List, Tuple
from ml.ner.extractor import CareerNERExtractor

class KnowledgeGraphBuilder:
    """
    Builds knowledge graphs from career entities.
    Nodes: skills, roles, tools, education, companies
    Edges: relationships with weights
    """
    
    def __init__(self):
        self.ner_extractor = CareerNERExtractor()
        
        # Predefined skill relationships for edge creation
        self.skill_relationships = {
            "Python": ["Django", "Flask", "FastAPI", "TensorFlow", "PyTorch"],
            "JavaScript": ["React", "Angular", "Vue", "Node.js", "Express"],
            "Java": ["Spring", "Hibernate", "Maven", "Gradle"],
            "AWS": ["EC2", "S3", "Lambda", "RDS", "DynamoDB"],
            "Docker": ["Kubernetes", "Jenkins", "CI/CD"],
            "SQL": ["MySQL", "PostgreSQL", "Oracle", "SQL Server"]
        }
    
    def build_graph(self, text: str, graph_type: str = "resume"):
        """
        Build knowledge graph from text.
        Returns a NetworkX DiGraph if networkx is available, or a plain dict otherwise.
        """
        # Extract structured entities
        structured_data = self.ner_extractor.extract_structured_data(text)

        if not _NX_AVAILABLE:
            # Fallback: return plain dict compatible with careerlens_engine expectations
            nodes = []
            for skill in structured_data["skills"]:
                nodes.append({"id": skill["name"], "type": "skill", "weight": skill["confidence"]})
            for role in structured_data["roles"]:
                nodes.append({"id": role["title"], "type": "role", "weight": role["confidence"]})
            for company in structured_data["companies"]:
                nodes.append({"id": company["name"], "type": "company", "weight": 0.7})
            return {"nodes": nodes, "edges": [], "type": graph_type}
        
        # Full NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        self._add_skill_nodes(G, structured_data["skills"])
        self._add_role_nodes(G, structured_data["roles"])
        self._add_tool_nodes(G, structured_data["tools"])
        self._add_education_nodes(G, structured_data["education"])
        self._add_company_nodes(G, structured_data["companies"])
        
        # Add edges based on relationships
        self._add_skill_relationships(G)
        self._add_role_skill_edges(G, structured_data)
        self._add_experience_edges(G, structured_data["experience_years"])
        
        # Add graph-level attributes
        G.graph["type"] = graph_type
        G.graph["node_count"] = G.number_of_nodes()
        G.graph["edge_count"] = G.number_of_edges()
        
        return G
    
    def _add_skill_nodes(self, G: nx.DiGraph, skills: List[Dict]):
        """Add skill nodes with attributes."""
        for skill in skills:
            node_id = f"skill_{skill['name'].lower().replace(' ', '_')}"
            G.add_node(node_id, 
                      type="SKILL",
                      name=skill['name'],
                      category=skill.get('category', 'GENERAL'),
                      confidence=skill['confidence'],
                      importance=self._calculate_skill_importance(skill['name']))
    
    def _add_role_nodes(self, G: nx.DiGraph, roles: List[Dict]):
        """Add role/job title nodes."""
        for role in roles:
            node_id = f"role_{role['title'].lower().replace(' ', '_')}"
            G.add_node(node_id,
                      type="ROLE", 
                      title=role['title'],
                      confidence=role['confidence'],
                      seniority=self._extract_seniority(role['title']))
    
    def _add_tool_nodes(self, G: nx.DiGraph, tools: List[Dict]):
        """Add tool/technology nodes."""
        for tool in tools:
            node_id = f"tool_{tool['name'].lower().replace(' ', '_')}"
            G.add_node(node_id,
                      type="TOOL",
                      name=tool['name'], 
                      confidence=tool['confidence'])
    
    def _add_education_nodes(self, G: nx.DiGraph, education: List[Dict]):
        """Add education nodes."""
        for edu in education:
            node_id = f"edu_{edu['institution'].lower().replace(' ', '_')}"
            G.add_node(node_id,
                      type="EDUCATION",
                      institution=edu['institution'],
                      confidence=edu['confidence'])
    
    def _add_company_nodes(self, G: nx.DiGraph, companies: List[Dict]):
        """Add company nodes."""
        for company in companies:
            node_id = f"company_{company['name'].lower().replace(' ', '_')}"
            G.add_node(node_id,
                      type="COMPANY",
                      name=company['name'],
                      confidence=company['confidence'])
    
    def _add_skill_relationships(self, G: nx.DiGraph):
        """Add edges between related skills."""
        skill_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'SKILL']
        
        for node in skill_nodes:
            skill_name = G.nodes[node]['name']
            
            # Check for predefined relationships
            if skill_name in self.skill_relationships:
                related_skills = self.skill_relationships[skill_name]
                
                for related in related_skills:
                    related_node = f"skill_{related.lower().replace(' ', '_')}"
                    if related_node in G.nodes:
                        G.add_edge(node, related_node,
                                  type="RELATED_TO",
                                  weight=0.8,
                                  relationship="prerequisite")
    
    def _add_role_skill_edges(self, G: nx.DiGraph, structured_data: Dict):
        """Add edges from roles to required skills."""
        role_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'ROLE']
        skill_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'SKILL']
        
        # Simple heuristic: connect roles to skills based on co-occurrence
        for role_node in role_nodes:
            for skill_node in skill_nodes:
                # Add edge with weight based on skill importance
                skill_importance = G.nodes[skill_node]['importance']
                G.add_edge(role_node, skill_node,
                          type="REQUIRES",
                          weight=skill_importance,
                          mandatory=skill_importance > 0.7)
    
    def _add_experience_edges(self, G: nx.DiGraph, experience_years: List[Dict]):
        """Add experience-based edge weights."""
        for exp in experience_years:
            years = exp.get('years', 0)
            # Boost weights for nodes based on experience
            for node in G.nodes():
                if G.nodes[node]['type'] in ['SKILL', 'TOOL']:
                    if 'weight' not in G.nodes[node]:
                        G.nodes[node]['weight'] = 1.0
                    G.nodes[node]['weight'] *= (1 + years * 0.1)  # Experience multiplier
    
    def _calculate_skill_importance(self, skill_name: str) -> float:
        """Calculate importance score for a skill (0-1)."""
        # High-demand skills get higher importance
        high_demand_skills = {
            "Python": 0.95, "JavaScript": 0.90, "React": 0.85, "AWS": 0.90,
            "Docker": 0.80, "Kubernetes": 0.85, "Machine Learning": 0.90,
            "SQL": 0.85, "Git": 0.75, "Linux": 0.70
        }
        
        return high_demand_skills.get(skill_name, 0.5)
    
    def _extract_seniority(self, role_title: str) -> str:
        """Extract seniority level from role title."""
        title_lower = role_title.lower()
        
        if any(word in title_lower for word in ['senior', 'sr', 'lead', 'principal']):
            return "senior"
        elif any(word in title_lower for word in ['junior', 'jr', 'entry', 'associate']):
            return "junior"
        else:
            return "mid"
    
    def get_graph_features(self, G) -> Dict:
        """Extract graph-level features for ML models."""
        if not _NX_AVAILABLE or not hasattr(G, 'number_of_nodes'):
            return {"node_count": 0, "edge_count": 0, "density": 0}
        avg_imp = 0.0
        if _NP_AVAILABLE:
            skill_importances = [d.get('importance', 0.5) for n, d in G.nodes(data=True) if d.get('type') == 'SKILL']
            avg_imp = float(np.mean(skill_importances)) if skill_importances else 0.5
        return {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "density": nx.density(G),
            "avg_clustering": nx.average_clustering(G.to_undirected()),
            "skill_count": len([n for n, d in G.nodes(data=True) if d.get('type') == 'SKILL']),
            "role_count": len([n for n, d in G.nodes(data=True) if d.get('type') == 'ROLE']),
            "tool_count": len([n for n, d in G.nodes(data=True) if d.get('type') == 'TOOL']),
            "avg_skill_importance": avg_imp
        }

# Usage example
if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    
    resume_text = """
    Senior Software Engineer with 5 years of experience in Python and Django.
    Built REST APIs and deployed on AWS. Experience with React, PostgreSQL, 
    and Docker. Worked at Google. B.Tech from IIT Delhi.
    """
    
    jd_text = """
    Looking for a Software Engineer with Python, Django, React experience.
    Must have AWS cloud experience and Docker knowledge. 3+ years required.
    """
    
    # Build graphs
    resume_graph = builder.build_graph(resume_text, "resume")
    jd_graph = builder.build_graph(jd_text, "job_description")
    
    print(f"Resume Graph: {resume_graph.number_of_nodes()} nodes, {resume_graph.number_of_edges()} edges")
    print(f"JD Graph: {jd_graph.number_of_nodes()} nodes, {jd_graph.number_of_edges()} edges")
    
    # Print nodes by type
    for graph_name, graph in [("Resume", resume_graph), ("JD", jd_graph)]:
        print(f"\n{graph_name} Nodes:")
        for node, data in graph.nodes(data=True):
            print(f"  {data['type']}: {data.get('name', data.get('title', node))}")