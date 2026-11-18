"""Known-role GraphSAGE research explorer, never an individual suitability prediction."""
import json
import hashlib
from pathlib import Path
from threading import Lock
from ml.model_registry import artifact_status

_lock = Lock()
_loaded = None


def catalog():
    root = Path('Datasets/esco')
    manifest = json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    if hashlib.sha256((root/'occupations.json').read_bytes()).hexdigest() != manifest['catalog_sha256']:
        raise ValueError('ESCO catalog checksum mismatch. Reacquire the public catalog.')
    return json.loads((root/'occupations.json').read_text(encoding='utf-8'))


def search_roles(query):
    return [row for row in catalog() if query.casefold() in row['title'].casefold()][:30]


def explore(code):
    status = artifact_status('ml/models/career', 'career_trajectory')
    if not status.get('trained'):
        return {'available':False, 'reason':status['reason']}
    names = {row['code']:row for row in catalog()}
    if code not in names:
        raise ValueError('Select a role from the local ESCO catalog.')
    global _loaded
    with _lock:
        import torch
        torch.set_num_threads(4)
        if _loaded is None:
            from ml.trajectory.link_predictor import CareerGraphSAGE
            saved = torch.load('ml/models/career/model.pt', map_location='cpu', weights_only=True)
            model = CareerGraphSAGE(**status['provenance']['model_config'])
            model.load_state_dict(saved['model_state_dict']); model.eval()
            _loaded = saved, model
        saved, model = _loaded
        if code not in saved['roles']:
            return {'available':False, 'reason':'This role has insufficient observations in the trained graph. Use the local evidence-linked career exploration instead.'}
        source = saved['roles'].index(code)
        with torch.inference_mode():
            sources = torch.full((len(saved['roles']),),source,dtype=torch.long)
            scores = model.predict_link(saved['features'], saved['edge_index'], sources, torch.arange(len(saved['roles']))).tolist()
        ranked = sorted([i for i in range(len(scores)) if i!=source and saved['roles'][i] in names],key=lambda i:(-scores[i],saved['roles'][i]))[:5]
        return {'available':True, 'current_role':names[code],
                'roles':[dict(names[saved['roles'][i]], activation=round(scores[i],6)) for i in ranked],
                'note':'Research ranking from historical Flemish transitions, conditioned only on your explicitly selected role. It does not use candidate skills or predict suitability, confidence, salary, or timing. Its held-out ranking was worse than the frequency baseline; use evidence-linked Gemma career ideas for exploration.',
                'evaluation':status['provenance']['evaluation']['test']}
