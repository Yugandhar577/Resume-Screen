"""Reproduce fitted weak wording models on source-disjoint fictional examples."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from data.label_aggregation import SnorkelAggregator
from training.validate_reviewed_data import validate_manifest
from ml.weak_wording import PATTERNS, votes, posterior


def main():
    root = Path('data/review_examples/ai-approved')
    provenance = validate_manifest(root/'jobs-reviewed.provenance.json','job_bias',root/'jobs-reviewed.csv',allow_ai_reviewed_synthetic=True)
    frame = pd.read_csv(root/'jobs-reviewed.csv')
    train,rest = next(GroupShuffleSplit(n_splits=1,train_size=.6,random_state=42).split(frame,groups=frame.source_id))
    val,test = next(GroupShuffleSplit(n_splits=1,train_size=.5,random_state=42).split(frame.iloc[rest],groups=frame.iloc[rest].source_id))
    splits = {'train':train,'val':rest[val],'test':rest[test]}
    arrays = {'prior':np.array([.5,.5])}; evaluation = {'val':{},'test':{}}
    for label in PATTERNS:
        matrix = np.array([votes(text,label) for text in frame.text])
        model = SnorkelAggregator().fit(matrix[train],['literal wording','word cluster','qualifier combination'])
        conditional = model.model.get_conditional_probs();arrays[label]=conditional
        # Assert safe numeric serialization reproduces the fitted implementation.
        expected = model.predict(matrix)
        actual = [posterior(conditional,arrays['prior'],row) for row in matrix]
        if any((a is None)!=(b is None) for a,b in zip(expected,actual)) or not np.allclose([a for a in expected if a is not None],[a for a in actual if a is not None]):
            raise ValueError('Serialized label model inference differs from Snorkel.')
        for split in ('val','test'):
            all_ids=splits[split];ids=[i for i in all_ids if actual[i] is not None];probability=np.array([actual[i] for i in ids]);truth=frame.iloc[ids][label]
            evaluation[split][label]={'rows':len(ids),'abstained':len(all_ids)-len(ids),'auc':float(roc_auc_score(truth,probability)) if truth.nunique()==2 else None,
                                    'f1':float(f1_score(truth,probability>=.5)), 'brier':float(brier_score_loss(truth,probability)), 'calibrated':False}
    output=Path('ml/models/weak_labels');output.mkdir(parents=True,exist_ok=True)
    checkpoint=output/'conditional.npz';np.savez(checkpoint,**arrays)
    manifest={'version':'snorkel-fictional-v1','task':'job_weak_labels','training_completed':True,'deployment_ready':False,
              'deployment_reason':'Correlated template rules and synthetic holdouts do not establish real-world validity.',
              'checkpoint':checkpoint.name,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'source':provenance,'patterns':PATTERNS,'evaluation':evaluation,
              'vote_code_sha256':hashlib.sha256(Path('ml/weak_wording.py').read_bytes()).hexdigest(),
              'evaluation_protocol':'Rule patterns were authored after inspecting the fictional examples; source-disjoint evaluation is a pipeline check, not a blind generalization test.',
              'split':{name:len(ids) for name,ids in splits.items()},
              'limitations':['Only 60 templated fictional descriptions, with phrasing repeated across source splits.','Correlated labeling functions violate independence assumptions.','Unsupported salary category omitted.']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(evaluation,indent=2))


if __name__=='__main__':main()
