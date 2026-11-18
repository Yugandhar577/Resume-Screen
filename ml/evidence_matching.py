"""Local pretrained semantic passage retrieval with exact evidence, across occupations."""
from pathlib import Path
import re
from threading import Lock

_lock=Lock()
_encoder=None


def status():
    import importlib.util
    return {'available':Path('Datasets/models/minilm/modules.json').is_file() and all(importlib.util.find_spec(name) for name in ('torch','sentence_transformers')),
            'method':'Pinned local semantic passage retrieval','predicts_outcomes':False}


def analyze(resume,jd):
    if not status()['available']:
        return {'available':False,'reason':'Local sentence encoder is not installed; baseline evidence checks remain available.'}
    def passages(text):
        result=[]
        for line in re.split(r'\n+|(?<=[.!?])\s+',text):
            if not line.strip() or '@' in line:continue
            for start in range(0,len(line),400):
                quote=line[start:start+400].strip()
                if quote:result.append(quote)
        return list(dict.fromkeys(result))
    resume_passages=passages(resume)[:32];job_passages=passages(jd)[:16]
    if not resume_passages or not job_passages:return {'available':False,'reason':'No usable work/requirement passages for semantic retrieval.'}
    global _encoder
    with _lock:
        import numpy as np
        if _encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(4)
            _encoder=SentenceTransformer('Datasets/models/minilm',local_files_only=True,trust_remote_code=False,device='cpu')
        vectors=_encoder.encode(resume_passages+job_passages,normalize_embeddings=True,show_progress_bar=False)
        similarities=vectors[len(resume_passages):]@vectors[:len(resume_passages)].T
        items=[]
        for i,quote in enumerate(job_passages):
            indices=np.argsort(-similarities[i])[:2]
            items.append({'job_quote':quote,'resume_passages':[{'quote':resume_passages[int(j)],'similarity':round(float(similarities[i,j]),4),
                'qualifier_review':bool(re.search(r'\b(?:no|not|never|without|lack|basic|beginner|learning|familiar)\b',resume_passages[int(j)],re.I))} for j in indices]})
        return {'available':True,'method':'Pinned local MiniLM semantic passage retrieval','items':items,
                'source_revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
                'note':'Closest source passages, not proof that a requirement is met. Review negation, proficiency, experience and education. At most 32 resume and 16 job passages of 400 characters are considered; long passages may reach the encoder token limit. Similarity is not hiring probability. No occupation-title allowlist or synthetic GAT score is required.'}
