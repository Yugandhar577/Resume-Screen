"""Local, versioned semantic graph inference. Never downloads during requests."""
import hashlib
import json
import re
import importlib.util
from pathlib import Path
from threading import Lock

MINILM_REPO = 'sentence-transformers/all-MiniLM-L6-v2'
MINILM_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
PUBLIC_ROLE_FAMILIES = ['Account Executive', 'Associate Product Manager', 'BI Analyst', 'Backend Engineer',
    'Business Analyst', 'Business Development Manager', 'Content Marketer', 'Customer Success Associate',
    'Customer Support Specialist', 'Data Analyst', 'FP&A Analyst', 'Financial Analyst', 'Full Stack Engineer',
    'Junior Accountant', 'Marketing Manager', 'Operations Manager', 'Performance Marketer', 'Product Manager',
    'Program Coordinator', 'Project Manager', 'Sales Representative', 'Software Engineer',
    'Technical Product Manager', 'Technical Support Specialist']


class SemanticGraphEncoder:
    def __init__(self, encoder_path):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(encoder_path), local_files_only=True, trust_remote_code=False, device='cpu')

    def graph(self, text):
        import torch
        from torch_geometric.data import Data
        lines = [line.strip() for line in text.splitlines() if line.strip() and '@' not in line]
        # Explicitly bounded, observed text nodes. Attention is not causal attribution.
        chunks = [line[start:start + 400] for line in lines for start in range(0, len(line), 400)][:32]
        if not chunks:
            raise ValueError('No usable text for semantic encoding.')
        features = self.model.encode(chunks, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        edges = [[i, i + 1] for i in range(len(chunks) - 1)] + [[i + 1, i] for i in range(len(chunks) - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        data = Data(x=features.cpu(), edge_index=edge_index)
        data.source_chunks = chunks
        return data


class SemanticMatcher:
    def __init__(self, directory='ml/models/gat_experimental'):
        self.directory = Path(directory)
        self.model = None
        self.encoder = None
        self.lock = Lock()
        self.metadata = None
        self.reason = 'No evaluated semantic checkpoint is installed.'
        manifest = self.directory / 'manifest.json'
        if manifest.is_file():
            try:
                metadata = json.loads(manifest.read_text(encoding='utf-8'))
                required = {'version', 'training_completed', 'checkpoint_sha256', 'evaluation', 'model_config', 'encoder_path', 'dataset'}
                if not required.issubset(metadata) or not metadata['training_completed'] or not metadata['evaluation'].get('test'):
                    raise ValueError('Checkpoint is missing training/evaluation provenance.')
                checkpoint = self.directory / 'model.pt'
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata['checkpoint_sha256']:
                    raise ValueError('Checkpoint checksum does not match its manifest.')
                if not Path(metadata['encoder_path']).is_dir():
                    raise ValueError('The local sentence encoder is missing.')
                self.metadata = metadata
                self.reason = None
            except Exception as exc:
                self.reason = str(exc)

    def status(self):
        dependencies = all(importlib.util.find_spec(name) for name in ('torch', 'torch_geometric', 'sentence_transformers'))
        return {'available': self.metadata is not None and dependencies,
                'reason': self.reason if dependencies else 'Optional semantic dependencies are not installed.',
                'version': self.metadata['version'] if self.metadata else None,
                'experimental': bool(self.metadata and self.metadata.get('experimental', True)),
                'dataset': self.metadata.get('dataset') if self.metadata else None}

    def analyze(self, resume, jd):
        if not self.status()['available']:
            return {'available': False, 'reason': self.status()['reason']}
        families = self.metadata.get('supported_role_families', [])
        if not any(re.search(r'(?<!\w)' + re.escape(role) + r'(?!\w)', jd, re.I) for role in families):
            return {'available': False, 'reason': 'No explicitly supported job-title family was found. This experiment does not cover every occupation; baseline evidence remains available.',
                    'supported_role_families': families}
        import torch
        from ml.gat.model import SiameseGAT
        with self.lock, torch.inference_mode():
            if self.model is None:
                torch.set_num_threads(4)
                self.encoder = SemanticGraphEncoder(self.metadata['encoder_path'])
                self.model = SiameseGAT(self.metadata['model_config'])
                checkpoint = torch.load(self.directory / 'model.pt', map_location='cpu', weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
            resume_graph = self.encoder.graph(resume)
            jd_graph = self.encoder.graph(jd)
            _, _, score = self.model(resume_graph, jd_graph)
            attention = {}
            for name, graph in [('resume', resume_graph), ('job', jd_graph)]:
                self.model.get_embeddings(graph)
                edges, weights = self.model.encoder.last_attention_weights
                outgoing = torch.zeros(graph.x.size(0))
                outgoing.scatter_add_(0, edges[0], weights.mean(1))
                attention[name] = [{'quote': chunk, 'attention': float(outgoing[index])}
                                   for index, chunk in enumerate(graph.source_chunks)]
            return {'available': True, 'similarity': round(float(score.item()), 4),
                    'method': 'MiniLM → observed text graph → trained Siamese GAT',
                    'provenance': self.metadata, 'attention': attention,
                    'note': 'Experimental synthetic-label similarity, not a hiring probability. Attention is observed model behavior, not a causal explanation. At most 32 text chunks are encoded.'}
