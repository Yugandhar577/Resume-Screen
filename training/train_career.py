"""Person-disjoint GraphSAGE benchmark; deliberately not deployed to individuals."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from training.environment import environment


def quarter(value):
    match = re.fullmatch(r'Q([1-4]) (\d{4})', str(value))
    return int(match[2]) * 4 + int(match[1]) if match else None


def transitions(frame, limit):
    people = sorted(frame['resume_id'].unique())
    random.Random(42).shuffle(people)
    selected = set(people[:limit])
    pairs = []
    for _, rows in frame[frame.resume_id.isin(selected)].groupby('resume_id', sort=False):
        dated = []
        for row in rows.itertuples():
            date = quarter(row.start_date)
            end = quarter(row.end_date)
            code = str(row.matched_code)
            if date is not None and end is not None and date <= end and code not in ('unknown', 'nan', 'None'):
                dated.append((date, end, code))
        dated.sort()
        # Only adjacent, non-overlapping experiences; simultaneous work is not a transition.
        for first, second in zip(dated, dated[1:]):
            if first[1] <= second[0] and first[2] != second[2]:
                pairs.append((first[2], second[2]))
    return pairs, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Datasets/jobhop')
    parser.add_argument('--output', default='ml/models/career')
    parser.add_argument('--people', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()
    if not 100 <= args.people <= 100000 or not 1 <= args.epochs <= 500:
        raise ValueError('Use 100–100000 people and 1–500 epochs.')
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    from ml.trajectory.link_predictor import CareerGraphSAGE
    torch.set_num_threads(4); torch.manual_seed(42); random.seed(42)
    root = Path(args.data)
    provenance = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
    raw, people = {}, {}
    for split in ('train', 'val', 'test'):
        frame = pd.read_parquet(root / f'JobHop_v2_{split}.parquet')
        raw[split], people[split] = transitions(frame, args.people if split == 'train' else max(100, args.people // 5))
        print(f'{split}: {len(people[split])} people, {len(raw[split])} adjacent transitions', flush=True)
    if any(people[a] & people[b] for a, b in [('train','val'), ('train','test'), ('val','test')]):
        raise ValueError('Person IDs overlap across source splits; stop rather than leak trajectories.')
    counts = Counter(raw['train'])
    frequency = Counter(code for pair in raw['train'] for code in pair)
    roles = sorted(code for code, count in frequency.items() if count >= 5)
    ids = {code: index for index, code in enumerate(roles)}
    pairs = {split: [(ids[a], ids[b]) for a,b in rows if a in ids and b in ids] for split, rows in raw.items()}
    if len(roles) < 5 or any(len(rows) < 10 for rows in pairs.values()):
        raise ValueError('Insufficient known-role transitions for evaluation.')
    config = {'input_dim': 64, 'hidden_dim': 32, 'output_dim': 16, 'num_layers': 2, 'dropout': 0.1}
    features = np.zeros((len(roles), 64), dtype=np.float32)
    for code, index in ids.items():
        for prefix in [code[:n] for n in range(1, len(code)+1)]:
            features[index, int(hashlib.sha256(prefix.encode()).hexdigest()[:8], 16) % 62] += 1
        features[index, 62] = np.log1p(frequency[code]) / 10
        features[index, 63] = 1
    x = torch.from_numpy(features)
    edges = sorted(set(pairs['train']))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    model = CareerGraphSAGE(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    rng = random.Random(42)
    samples = {}
    for split, positive in pairs.items():
        observed = set(positive) | set(pairs['train'])
        negatives = []
        for source, _ in positive:
            candidates = [target for target in range(len(roles)) if target != source and (source, target) not in observed]
            if candidates:
                negatives.append((source, rng.choice(candidates)))
        joined = positive + negatives
        samples[split] = (torch.tensor(joined).t(), torch.tensor([1.] * len(positive) + [0.] * len(negatives)))
    def scores(split):
        model.eval()
        with torch.inference_mode():
            src, dst = samples[split][0]
            return model.predict_link(x, edge_index, src, dst).numpy()
    best, best_auc = None, -1
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad()
        (src, dst), labels = samples['train']
        predictions = model.predict_link(x, edge_index, src, dst)
        loss = torch.nn.functional.binary_cross_entropy(predictions, labels)
        loss.backward(); optimizer.step()
        auc = roc_auc_score(samples['val'][1], scores('val'))
        if auc > best_auc:
            best_auc = auc; best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f'epoch={epoch+1} loss={loss.item():.4f} validation_auc={auc:.4f}', flush=True)
    model.load_state_dict(best)
    validation = scores('val')
    threshold = max(np.linspace(0, 1, 101), key=lambda t:f1_score(samples['val'][1], validation >= t, zero_division=0))
    baseline_counts = Counter(pairs['train'])
    def evaluate(split):
        predicted = scores(split); labels = samples[split][1].numpy()
        baseline = np.array([baseline_counts[tuple(pair)] for pair in samples[split][0].t().tolist()], dtype=float)
        recall, reciprocal, baseline_recall, baseline_rr = [], [], [], []
        model.eval()
        with torch.inference_mode():
            embeddings = model(x, edge_index)
            for source in sorted(set(a for a,b in pairs[split])):
                destinations = torch.arange(len(roles))
                values = model.link_predictor(torch.cat([embeddings[source].expand(len(roles), -1), embeddings[destinations]], dim=1)).squeeze().numpy()
                values[source] = -1
                ranking = sorted(range(len(roles)), key=lambda i:(-values[i], roles[i]))
                base_ranking = sorted(range(len(roles)), key=lambda i:(-baseline_counts[(source,i)], roles[i]))
                for _, target in [pair for pair in pairs[split] if pair[0] == source]:
                    rank = ranking.index(target)+1; base_rank = base_ranking.index(target)+1
                    recall.append(rank <= 5); reciprocal.append(1/rank)
                    baseline_recall.append(base_rank <= 5); baseline_rr.append(1/base_rank)
        return {'pairs': len(labels), 'positive_transitions': len(pairs[split]), 'threshold': float(threshold),
                'auc': float(roc_auc_score(labels, predicted)), 'average_precision': float(average_precision_score(labels, predicted)),
                'f1': float(f1_score(labels, predicted >= threshold)), 'brier': float(np.mean((predicted-labels)**2)), 'calibrated': False,
                'recall_at_5': float(np.mean(recall)), 'mrr': float(np.mean(reciprocal)),
                'frequency_baseline': {'auc': float(roc_auc_score(labels, baseline)), 'recall_at_5': float(np.mean(baseline_recall)), 'mrr': float(np.mean(baseline_rr))},
                'excluded_unknown_or_infrequent_role_transitions': len(raw[split])-len(pairs[split]), 'candidate_occupations': len(roles)}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'model.pt'
    torch.save({'model_state_dict': model.state_dict(), 'roles': roles, 'features': x, 'edge_index': edge_index}, checkpoint)
    metadata = {'version': 'graphsage-jobhop-research-v1', 'task': 'career_trajectory', 'training_completed': True, 'environment': environment(),
                'deployment_ready': False, 'deployment_reason': 'No fairness or cross-market validation; source warns against individual deployment without these checks. Public data does not contain candidate skills.',
                'checkpoint': 'model.pt', 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'model_config': config, 'seed': 42, 'epochs': args.epochs, 'source': provenance,
                'split': 'Provided person-disjoint train/validation/test; deterministic person subsampling, training-only graph and role vocabulary.',
                'evaluation': {split:evaluate(split) for split in ('val','test')},
                'limitations': ['Flemish public-employment-service population, not representative of India or all workers.',
                                'LLM-extracted records and proprietary ESCO assignment are not error-free.',
                                'Known-role transition links can recur across people; not an unseen-edge benchmark.',
                                'Uniform sampled negatives are not known impossible career moves.',
                                'No skills, calibrated timelines, protected attributes, or individual suitability conclusions.']}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata['evaluation'], indent=2))


if __name__ == '__main__':
    main()
