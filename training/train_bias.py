"""Review-gated BERT training with source-disjoint holdouts; never fake fairness scores."""
import argparse
from pathlib import Path
import hashlib
import json
from training.validate_reviewed_data import validate_manifest
from training.environment import environment

ALLOWED_LABELS = {'Vague', 'Biased', 'Salary_Mismatched', 'Experience_Inflated'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Reviewed CSV: source_id,text, and explicitly supported binary label columns')
    parser.add_argument('--provenance', required=True, help='Reviewed manifest with labels, source/license, reviewer, and data_sha256')
    parser.add_argument('--output', default='ml/models/jd_bias')
    parser.add_argument('--allow-ai-reviewed-synthetic', action='store_true', help='Explicit experimental mode; never constitutes human review')
    args = parser.parse_args()
    provenance = validate_manifest(args.provenance, 'job_bias', args.data, args.allow_ai_reviewed_synthetic)
    labels = provenance.get('labels', [])
    if not labels or not set(labels).issubset(ALLOWED_LABELS) or len(set(labels)) != len(labels):
        raise ValueError('Manifest must name the label columns supported by the reviewed data.')
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss
    data = pd.read_csv(args.data, keep_default_na=False)
    if not {'source_id','text',*labels}.issubset(data.columns) or data.source_id.nunique() < 50:
        raise ValueError('Use at least 50 distinct reviewed source documents with text and supported label columns.')
    if data.text.str.strip().eq('').any() or not np.isin(data[labels].to_numpy(), [0,1]).all():
        raise ValueError('Text must be nonempty and reviewed labels must be binary.')
    data = data.drop_duplicates(subset=['text']).reset_index(drop=True)
    training, rest = next(GroupShuffleSplit(n_splits=1, train_size=.6, random_state=42).split(data, groups=data.source_id))
    validation, test = next(GroupShuffleSplit(n_splits=1, train_size=.5, random_state=42).split(data.iloc[rest], groups=data.iloc[rest].source_id))
    frames = {'train':data.iloc[training], 'val':data.iloc[rest[validation]], 'test':data.iloc[rest[test]]}
    if any(rows[label].nunique() < 2 for rows in frames.values() for label in labels):
        raise ValueError('Each supported label needs both classes in every holdout. Collect more reviewed data or revise grouping.')
    import torch
    torch.set_num_threads(4); torch.manual_seed(42)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
    base = 'google-bert/bert-base-uncased'
    revision = '86b5e0934494bd15c9632b12f734a8a67f723594'
    # Explicit training downloads only; no remote code is allowed.
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(base, revision=revision, num_labels=len(labels), problem_type='multi_label_classification',
                                                               id2label=dict(enumerate(labels)), label2id={name:i for i,name in enumerate(labels)})
    class ReviewedDataset(torch.utils.data.Dataset):
        def __init__(self, rows): self.rows = rows.reset_index(drop=True)
        def __len__(self): return len(self.rows)
        def __getitem__(self, index):
            row = self.rows.iloc[index]
            encoded = tokenizer(row.text, truncation=True, max_length=512)
            return {**encoded, 'labels':[float(row[label]) for label in labels]}
    datasets = {name:ReviewedDataset(rows) for name,rows in frames.items()}
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=args.output, num_train_epochs=3, seed=42,
                      per_device_train_batch_size=2, per_device_eval_batch_size=2, learning_rate=2e-5,
                      save_strategy='epoch', eval_strategy='epoch', load_best_model_at_end=True, report_to=[], save_total_limit=2),
                      train_dataset=datasets['train'], eval_dataset=datasets['val'], processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train()
    val_probability = torch.sigmoid(torch.from_numpy(trainer.predict(datasets['val']).predictions)).numpy()
    thresholds = {label:float(max(np.linspace(0,1,101), key=lambda t:f1_score(frames['val'][label], val_probability[:,i]>=t, zero_division=0))) for i,label in enumerate(labels)}
    evaluation = {}
    for split in ('val','test'):
        probability = torch.sigmoid(torch.from_numpy(trainer.predict(datasets[split]).predictions)).numpy()
        evaluation[split] = {'rows':len(frames[split]), 'labels':{}, 'calibrated':False}
        for i,label in enumerate(labels):
            truth = frames[split][label]; decision = probability[:,i] >= thresholds[label]
            evaluation[split]['labels'][label] = {'auc':float(roc_auc_score(truth, probability[:,i])), 'f1':float(f1_score(truth, decision)),
                'precision':float(precision_score(truth, decision, zero_division=0)), 'recall':float(recall_score(truth, decision)),
                'brier':float(brier_score_loss(truth, probability[:,i])), 'validation_threshold':thresholds[label]}
    output = Path(args.output); trainer.save_model(output); tokenizer.save_pretrained(output)
    checkpoint = output/'model.safetensors'
    metadata = {'version':'bert-reviewed-jd-v1', 'task':'job_bias', 'training_completed':True, 'deployment_ready':False,
                'deployment_reason':'Phrase attribution and deployment-context evaluation are not yet validated.',
                'checkpoint':checkpoint.name, 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'base_model':base, 'base_revision':revision, 'base_commit':getattr(model.config, '_commit_hash', None), 'supported_labels':labels,
                'source':provenance, 'environment':environment(), 'evaluation':evaluation,
                'split':'Source-document-disjoint 60/20/20; exact duplicate texts removed',
                'limitations':['Raw activations are not calibrated fairness probabilities.', 'No overall quality score.', 'Unsupported categories are unavailable, not negative.', 'No candidate traits or hiring outcomes are inferred.']}
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
