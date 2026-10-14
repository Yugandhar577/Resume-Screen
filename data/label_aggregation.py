"""Actual Snorkel label aggregation; weak posteriors are never verified outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


class SnorkelAggregator:
    def __init__(self):
        self.model = None
        self.columns = None

    def fit(self, votes, columns, epochs=500):
        votes = np.asarray(votes)
        if votes.ndim != 2 or votes.shape[0] < 20 or votes.shape[1] < 3 or len(columns) != votes.shape[1]:
            raise ValueError('Use at least 20 examples and three documented labeling functions.')
        if not np.isin(votes, [-1, 0, 1]).all():
            raise ValueError('Votes must be -1 (abstain), 0, or 1.')
        if not {0, 1}.issubset(set(votes.flatten())):
            raise ValueError('Only one class was observed. A meaningful binary label model cannot be fitted.')
        from snorkel.labeling.model import LabelModel
        self.columns = list(columns)
        self.model = LabelModel(cardinality=2, verbose=False)
        self.model.fit(L_train=votes.astype(int), n_epochs=epochs, seed=42, log_freq=100)
        return self

    def predict(self, votes):
        if self.model is None:
            raise ValueError('Fit the label model first.')
        votes = np.asarray(votes)
        if votes.ndim != 2 or votes.shape[1] != len(self.columns) or not np.isin(votes, [-1,0,1]).all():
            raise ValueError('Vote shape or values do not match the trained labeling functions.')
        probabilities = self.model.predict_proba(votes.astype(int))[:,1]
        return [None if np.all(row == -1) else float(probability) for row,probability in zip(votes, probabilities)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--votes', required=True, help='CSV: source_id,text,lf_*; one explicitly defined binary wording task')
    parser.add_argument('--provenance', required=True, help='JSON: source_url,license,task,data_sha256,label_definitions')
    parser.add_argument('--output', default='Datasets/weak_labels')
    args = parser.parse_args()
    import pandas as pd
    provenance = json.loads(Path(args.provenance).read_text(encoding='utf-8'))
    required = {'source_url','license','task','data_sha256','label_definitions'}
    if not required.issubset(provenance) or hashlib.sha256(Path(args.votes).read_bytes()).hexdigest() != provenance['data_sha256']:
        raise ValueError('Document the source, license, task, labeling functions, and matching dataset checksum.')
    data = pd.read_csv(args.votes)
    columns = [name for name in data.columns if name.startswith('lf_')]
    if not {'source_id','text'}.issubset(data.columns):
        raise ValueError('CSV requires source_id,text and lf_* columns.')
    model = SnorkelAggregator().fit(data[columns].to_numpy(), columns)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    data['weak_probability'] = model.predict(data[columns].to_numpy())
    data.to_csv(output/'weak_labels.csv', index=False)
    # No unsafe pickle: preserve learned conditional probabilities as numeric arrays.
    np.savez(output/'conditional_probabilities.npz', conditional=model.model.get_conditional_probs())
    (output/'manifest.json').write_text(json.dumps({'source': provenance, 'seed':42, 'label_functions':columns,
        'method':'Fitted Snorkel LabelModel, not majority voting',
        'limitations':['Weak labels are approximations; evaluate against independent reviewed holdouts.',
                       'Correlated labeling functions can violate LabelModel assumptions.',
                       'All-abstain examples remain unlabeled; no salary or fairness conclusions without supporting data.']}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
