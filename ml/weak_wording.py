"""Documented rule votes and fitted Snorkel inference; never verified fairness labels."""
import re

PATTERNS = {
    'Vague': [r'rockstar|perfect culture fit|unspecified tasks', r'do everything|always go above|unlimited flexibility', r'rockstar|above and beyond|unspecified'],
    'Biased': [r'young candidates only|male candidates only|native english speakers only', r'\b(?:young|male|native english)\b', r'(?:candidates|speakers) only'],
    'Experience_Inflated': [r'entry.level.*12 years', r'no prior professional experience.*12 years', r'training role.*nevertheless.*12 years'],
}


def votes(text, label):
    known_structure = bool(re.search(r'responsibilities|requirements|entry.level|role|experience is preferred|equivalent demonstrated practice', text, re.I))
    return [1 if re.search(pattern, text, re.I|re.S) else (0 if known_structure else -1) for pattern in PATTERNS[label]]


def posterior(conditional, prior, row):
    import numpy as np
    if all(v == -1 for v in row):
        return None
    # Matches Snorkel's independent LF inference: abstains contribute no factor.
    log_scores = np.log(np.clip(prior,1e-12,1))
    for i,value in enumerate(row):
        if value != -1:
            log_scores += np.log(np.clip(conditional[i,value+1,:],1e-12,1))
    probabilities = np.exp(log_scores-log_scores.max())
    return float(probabilities[1]/probabilities.sum())


def analyze(text):
    from ml.model_registry import artifact_status
    from pathlib import Path
    import hashlib
    import numpy as np
    status = artifact_status('ml/models/weak_labels','job_weak_labels')
    if not status.get('trained'):
        return {'available':False, 'reason':status['reason']}
    if status['provenance'].get('vote_code_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        return {'available':False, 'reason':'Rule definitions changed since fitting. Refit the weak-label experiment.'}
    with np.load('ml/models/weak_labels/conditional.npz',allow_pickle=False) as data:
        labels = []
        for label in PATTERNS:
            row = votes(text,label)
            labels.append({'category':label, 'votes':row, 'weak_probability':posterior(data[label],data['prior'],row)})
    return {'available':True, 'labels':labels,
            'note':'Fitted Snorkel aggregation of three correlated pattern votes per task. Weak probabilities are experimental, uncalibrated approximations. All-abstain inputs stay unlabeled; no fairness certification.'}
