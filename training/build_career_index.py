"""Build a local frozen-encoder occupation index from public taxonomy text only."""
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    root=Path('Datasets/esco')
    source=json.loads((root/'profiles.provenance.json').read_text(encoding='utf-8'))
    raw=(root/'profiles.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=source['profiles_sha256']:raise ValueError('Profile checksum mismatch.')
    profiles=json.loads(raw)
    encoder=SentenceTransformer('Datasets/models/minilm',device='cpu',local_files_only=True,trust_remote_code=False)
    texts=[row['title']+'. '+row['description']+' Essential: '+', '.join(s['name'] for s in row['essential'])+' Optional: '+', '.join(s['name'] for s in row['optional']) for row in profiles]
    # Index text is explicitly bounded to the encoder's token limit by its tokenizer.
    # Full profiles remain available in responses; this is retrieval, not a qualifications score.
    embeddings=encoder.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True)
    target=root/'career_index.npz'
    np.savez_compressed(target,embeddings=embeddings,codes=np.array([row['code'] for row in profiles]))
    metadata={'method':'Frozen pinned MiniLM retrieval over ESCO occupation descriptions/skills',
              'encoder_revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41','encoder_max_tokens':encoder.max_seq_length,
              'profiles_sha256':source['profiles_sha256'],'index_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'rows':len(profiles),'source':source,'note':'Retrieval index; not a trained career-outcome predictor. Long taxonomy descriptions may be truncated by the documented encoder token limit.'}
    (root/'career_index.provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Indexed {len(profiles)} public career profiles.')


if __name__=='__main__':main()
