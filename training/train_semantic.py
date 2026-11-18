"""Reproducible GAT experiment on a licensed synthetic corpus, with disjoint splits."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from ml.gat.model import SiameseGAT, ContrastiveLoss
from ml.semantic_matching import SemanticGraphEncoder, MINILM_REPO, MINILM_REVISION
from ml.skills import extract_skills
from training.environment import environment


def document(row, kind):
    if kind == 'resume':
        return f"{row['role']}\n{row['summary']}\nSkills: {', '.join(row['skills'])}\n" + '\n'.join(row['experience_bullets'])
    return f"{row['job_title']}\n{row['description']}\nRequired skills: {', '.join(row['must_have_skills'])}\nPreferred: {', '.join(row['nice_to_have_skills'])}"


def evaluate(model, pairs, graphs, baseline=False, threshold=0.5, select_threshold=False, return_predictions=False):
    labels, scores, groups, candidates = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), 32):
            batch = pairs[start:start + 32]
            if baseline:
                values = []
                for resume_id, job_id, label in batch:
                    rs = set(extract_skills(graphs['texts'][resume_id]))
                    js = set(extract_skills(graphs['texts'][job_id]))
                    values.append(len(rs & js) / len(js) if js else 0)
            else:
                _, _, similarity = model(Batch.from_data_list([graphs[r] for r, j, y in batch]),
                                         Batch.from_data_list([graphs[j] for r, j, y in batch]))
                values = similarity.tolist()
            scores.extend(values); labels.extend(y for r, j, y in batch); groups.extend(j for r, j, y in batch)
            candidates.extend(r for r, j, y in batch)
    values = np.array(scores)
    if select_threshold:
        # Encode/predict once, then select on validation scores without 101 repeated model passes.
        candidates_thresholds = np.linspace(0 if baseline else -1, 1, 101)
        threshold = max(candidates_thresholds, key=lambda value: f1_score(labels, values >= value, zero_division=0))
    predicted = values >= threshold
    ranked = []
    for group in sorted(set(groups)):
        # Labels and insertion order must not resolve ties (positives are assembled first).
        rows = [(score, label, hashlib.sha256(f'{group}:{candidate}'.encode()).hexdigest())
                for score, label, key, candidate in zip(scores, labels, groups, candidates) if key == group]
        ranked.append(sorted(rows, key=lambda item: (-item[0], item[2])))
    recalls, reciprocal = [], []
    for rows in ranked:
        total = sum(label for score, label, tie in rows)
        recalls.append(sum(label for score, label, tie in rows[:5]) / total if total else 0)
        reciprocal.append(next((1 / (index + 1) for index, (score, label, tie) in enumerate(rows) if label), 0))
    # Cosine is mapped to [0,1] only for a reliability diagnostic; it is not calibrated probability.
    mapped = np.clip(values if baseline else (values + 1) / 2, 0, 1)
    report = {'pairs': len(pairs), 'jobs': len(ranked), 'threshold': float(threshold), 'accuracy': float(accuracy_score(labels, predicted)),
            'f1': float(f1_score(labels, predicted, zero_division=0)),
            'auc': float(roc_auc_score(labels, values)), 'recall_at_5': float(np.mean(recalls)),
            'mrr': float(np.mean(reciprocal)), 'brier_diagnostic': float(brier_score_loss(labels, mapped)),
            'calibrated': False, 'ranking_candidate_pool': 'sampled disjoint subset; not the full labor market'}
    return (report,np.array(labels),values) if return_predictions else report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--jobs', type=int, default=90)
    parser.add_argument('--resumes', type=int, default=600)
    parser.add_argument('--output', default='ml/models/gat_experimental')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--graph-mode', choices=['text','entity-v1'], default='text')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42); random.seed(42)
    root = Path('Datasets/candidate-matching-synthetic')
    provenance = json.loads((root / 'provenance.json').read_text())
    resumes = pd.read_parquet(root / 'resumes-train-00000-of-00001.parquet').sample(n=min(args.resumes, 10000), random_state=42)
    jobs = pd.read_parquet(root / 'jobs-train-00000-of-00001.parquet').sample(n=min(args.jobs, 2500), random_state=42)
    matches = pd.read_parquet(root / 'matches-train-00000-of-00001.parquet')
    relevant = {row.job_id: set(row.relevant_resume_ids) for row in matches.itertuples()}
    resumes = resumes.reset_index(drop=True); jobs = jobs.reset_index(drop=True)
    def split(frame, name):
        n = len(frame)
        return {'train': frame.iloc[:int(n*.6)][name].tolist(), 'val': frame.iloc[int(n*.6):int(n*.8)][name].tolist(),
                'test': frame.iloc[int(n*.8):][name].tolist()}
    resume_splits, job_splits = split(resumes, 'resume_id'), split(jobs, 'job_id')
    pairs = {}
    for key in ('train', 'val', 'test'):
        rows = []
        for job in job_splits[key]:
            positives = [r for r in resume_splits[key] if r in relevant[job]]
            negatives = [r for r in resume_splits[key] if r not in relevant[job]]
            for resume_id in positives:
                rows.append((resume_id, job, 1))
            for resume_id in random.sample(negatives, min(max(len(positives) * 3, 3), len(negatives))):
                rows.append((resume_id, job, 0))
        if not rows or len({label for r, j, label in rows}) < 2:
            raise ValueError(f'{key} split lacks both labels. Increase --resumes/--jobs.')
        pairs[key] = rows
    from sentence_transformers import SentenceTransformer
    encoder_path = Path('Datasets/models/minilm')
    if not (encoder_path / 'modules.json').is_file():
        encoder_path.mkdir(parents=True, exist_ok=True)
        print('Downloading pinned Apache-2.0 sentence encoder...', flush=True)
        SentenceTransformer(MINILM_REPO, revision=MINILM_REVISION, token=False, trust_remote_code=False).save(str(encoder_path))
    encoder = SemanticGraphEncoder(encoder_path, args.graph_mode)
    texts = {row['resume_id']: document(row, 'resume') for _, row in resumes.iterrows()}
    texts.update({row['job_id']: document(row, 'job') for _, row in jobs.iterrows()})
    graphs = {'texts': texts}
    print(f'Encoding {len(texts)} documents...', flush=True)
    for key, text in texts.items():
        graphs[key] = encoder.graph(text)
    config = {'input_dim': 384, 'hidden_dim': 32, 'output_dim': 32, 'num_heads': 2, 'num_layers': 2, 'dropout': .1}
    model = SiameseGAT(config); optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    criterion = ContrastiveLoss(temperature=.2)
    best, best_state = -1, None
    if args.evaluate_only:
        saved = torch.load(Path(args.output) / 'model.pt', map_location='cpu', weights_only=True)
        best_state = saved['model_state_dict']
    for epoch in range(0 if args.evaluate_only else args.epochs):
        model.train(); random.shuffle(pairs['train']); losses = []
        for start in range(0, len(pairs['train']), 32):
            batch = pairs['train'][start:start+32]
            left, right, _ = model(Batch.from_data_list([graphs[r] for r, j, y in batch]), Batch.from_data_list([graphs[j] for r, j, y in batch]))
            loss = criterion(left, right, torch.tensor([y for r, j, y in batch], dtype=torch.float))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        metrics = evaluate(model, pairs['val'], graphs)
        if metrics['auc'] > best:
            best = metrics['auc']; best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(f'Epoch {epoch+1}: loss={np.mean(losses):.4f}, validation AUC={metrics["auc"]:.4f}', flush=True)
    model.load_state_dict(best_state)
    # Choose a classification threshold on validation only; test is never used to tune it.
    validation_report,validation_labels,validation_scores = evaluate(model, pairs['val'], graphs, select_threshold=True, return_predictions=True)
    threshold = validation_report['threshold']
    test_report,test_labels,test_scores = evaluate(model, pairs['test'], graphs, threshold=threshold, return_predictions=True)
    report = {'val': validation_report, 'test': test_report}
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds='clip').fit(validation_scores,validation_labels)
    calibrated = calibrator.predict(test_scores)
    bins = []
    for start in np.linspace(0,.9,10):
        mask = (calibrated >= start) & (calibrated <= 1 if start >= .9 else calibrated < start+.1)
        if mask.any():
            bins.append({'count':int(mask.sum()),'mean_estimate':float(calibrated[mask].mean()),'observed_synthetic_rate':float(test_labels[mask].mean())})
    calibration = {'method':'Validation-only isotonic regression', 'x':calibrator.X_thresholds_.tolist(), 'y':calibrator.y_thresholds_.tolist(),
                   'test_brier':float(brier_score_loss(test_labels,calibrated)), 'test_reliability_bins':bins,
                   'test_ece':sum(b['count']*abs(b['mean_estimate']-b['observed_synthetic_rate']) for b in bins)/len(test_labels),
                   'note':'Calibration targets synthetic relevance labels and the sampled benchmark prevalence only. It is not a real-world relevance or hiring probability.'}
    report['baseline_val'] = evaluate(model, pairs['val'], graphs, baseline=True, select_threshold=True)
    baseline_threshold = report['baseline_val']['threshold']
    report['baseline_test'] = evaluate(model, pairs['test'], graphs, baseline=True, threshold=baseline_threshold)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'model.pt'; torch.save({'model_state_dict': model.state_dict()}, checkpoint)
    manifest = {'version': 'gat-semantic-synthetic-'+args.graph_mode+'-v1', 'graph_mode':args.graph_mode,
                'ner': {'package':'en_core_web_sm','version':'3.8.0','license':'MIT','type':'Pretrained general English NER plus canonical career vocabulary; not career-fine-tuned'} if args.graph_mode=='entity-v1' else None,
                'training_completed': True, 'experimental': True, 'environment': environment(),
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'model_config': config,
                'encoder_path': str(encoder_path), 'encoder': {'source': MINILM_REPO, 'revision': MINILM_REVISION, 'license': 'Apache-2.0'},
                'dataset': provenance, 'evaluation': report, 'calibration':calibration, 'seed': 42, 'epochs': args.epochs,
                'supported_role_families': sorted(resumes['role'].unique().tolist()),
                'classification_threshold': float(threshold), 'threshold_source': 'Validation F1 only',
                'subset': {'jobs': args.jobs, 'resumes': args.resumes}, 'ranking_ties': 'Label-independent SHA256 ordering of query and candidate IDs',
                'split': {'method': 'Both resume IDs and job IDs disjoint', 'counts': {key: len(value) for key,value in pairs.items()}},
                'limitations': ['Synthetic rule-generated relevance only', 'No evidence of real hiring validity', 'Not calibrated', 'Small sampled ranking pools']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
