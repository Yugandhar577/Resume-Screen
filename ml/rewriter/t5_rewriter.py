"""Reviewed-data T5 research pipeline; never reward invented metrics or skills."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from ml.skills import extract_skills

BASE_MODEL = 'google-t5/t5-small'
BASE_REVISION = 'df1b051c49625cf57a3d0d8d3863ed4d13564fe4'


def factual_checks(original, rewritten):
    """Necessary invariants, not sufficient semantic or identity validation."""
    numeric = lambda text: Counter(re.findall(r'\d+(?:[.,]\d+)*%?', text))
    negation = lambda text: Counter(re.findall(r'\b(?:not|never|without|no)\b', text.lower()))
    return {'numbers_preserved': numeric(original) == numeric(rewritten),
            'recognized_skills_preserved': extract_skills(original) == extract_skills(rewritten),
            'negation_preserved': negation(original) == negation(rewritten),
            'requires_human_review': True,
            'note': 'These checks cannot prove preservation of employers, ownership, dates, or role meaning. No arbitrary generated edit is automatically applied.'}


class ResumeRewriteDataset(Dataset):
    def __init__(self, pairs_df, tokenizer, max_length=256):
        self.pairs = pairs_df.reset_index(drop=True); self.tokenizer = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.pairs)
    def __getitem__(self, index):
        row = self.pairs.iloc[index]
        source = f"rewrite resume bullet for job: {row['jd_context']} original: {row['original']}"
        inputs = self.tokenizer(source, max_length=self.max_length, truncation=True)
        inputs['labels'] = self.tokenizer(str(row['improved']), max_length=self.max_length, truncation=True)['input_ids']
        return inputs


class T5ResumeRewriter:
    def __init__(self, model_name=BASE_MODEL, model_path=None):
        self.trained = False
        self.model_name = model_name
        if model_path:
            from ml.model_registry import artifact_status
            status = artifact_status(model_path, 'resume_rewriting')
            if not status.get('trained'):
                raise ValueError(status['reason'])
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
            self.trained = True
        else:
            # Explicit research/training download only; this class is never called by the API.
            options = {'revision': BASE_REVISION} if model_name == BASE_MODEL else {}
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False, **options)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name, **options)

    def train(self, training_pairs, save_path='ml/models/t5_rewriter', provenance=None):
        if not provenance or provenance.get('reviewed') is not True:
            raise ValueError('Documented licensed, human-reviewed provenance is required before training.')
        required = {'source_id', 'original', 'improved', 'jd_context'}
        if not required.issubset(training_pairs.columns):
            raise ValueError('CSV requires source_id, original, improved, jd_context; source_id identifies the original resume, not each bullet.')
        data = training_pairs.drop_duplicates(subset=['source_id', 'original', 'improved', 'jd_context'])
        sources = data.source_id.drop_duplicates().sample(frac=1, random_state=42).tolist()
        if len(sources) < 10:
            raise ValueError('At least ten distinct source resumes are required for separate train/validation/test sets.')
        count = max(1, len(sources)//5)
        split = {'val': set(sources[:count]), 'test': set(sources[count:count*2]), 'train': set(sources[count*2:])}
        frames = {name: data[data.source_id.isin(ids)].reset_index(drop=True) for name,ids in split.items()}
        originals = {name: set(rows.original) for name,rows in frames.items()}
        if any(originals[a] & originals[b] for a,b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError('Duplicated original bullets cross source splits. Group these sources before training.')
        datasets = {name: ResumeRewriteDataset(rows, self.tokenizer) for name,rows in frames.items()}
        arguments = TrainingArguments(output_dir=save_path, num_train_epochs=3, per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2, learning_rate=5e-5, weight_decay=0.01,
                                      save_strategy='epoch', eval_strategy='epoch', load_best_model_at_end=True,
                                      metric_for_best_model='eval_loss', greater_is_better=False, seed=42, report_to=[])
        trainer = Trainer(model=self.model, args=arguments, train_dataset=datasets['train'], eval_dataset=datasets['val'],
                          processing_class=self.tokenizer, data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model))
        trainer.train()
        self.trained = True
        evaluation = {'val': trainer.evaluate(datasets['val']), 'test': trainer.evaluate(datasets['test'])}
        checks = []
        for row in frames['test'].head(200).itertuples():
            generated = self.rewrite_bullet(row.original, row.jd_context, 1)[0]['rewritten']
            checks.append(factual_checks(row.original, generated))
        evaluation['test']['generation_examples'] = len(checks)
        for key in ('numbers_preserved', 'recognized_skills_preserved', 'negation_preserved'):
            evaluation['test'][key+'_rate'] = sum(item[key] for item in checks) / len(checks)
        output = Path(save_path); output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output, safe_serialization=True)
        self.tokenizer.save_pretrained(output)
        checkpoint = output / 'model.safetensors'
        metadata = {'version': 't5-reviewed-v1', 'task': 'resume_rewriting', 'training_completed': True,
                    'deployment_ready': False, 'deployment_reason': 'Generated outputs still require semantic factual and human writing-quality evaluation.',
                    'checkpoint': checkpoint.name, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    'source': provenance, 'base_model': self.model_name, 'base_revision': BASE_REVISION, 'seed': 42,
                    'split': {'method': 'Source-resume-disjoint, excluding duplicate original text', 'rows': {key:len(rows) for key,rows in frames.items()}},
                    'evaluation': evaluation, 'limitations': ['Invariant checks are necessary, not semantic guarantees.', 'No measured human writing-quality result yet.', 'No automatic addition of missing skills or metrics.']}
        (output/'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return metadata

    def rewrite_bullet(self, original_bullet, jd_context='', num_alternatives=3):
        if not self.trained:
            raise RuntimeError('No trained resume rewriter is loaded. Base T5 is not a resume rewriter.')
        if not 1 <= num_alternatives <= 3:
            raise ValueError('Request one to three alternatives.')
        inputs = self.tokenizer(f'rewrite resume bullet for job: {jd_context} original: {original_bullet}',
                                max_length=256, truncation=True, return_tensors='pt')
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=150, num_beams=max(2, num_alternatives),
                                          num_return_sequences=num_alternatives, do_sample=False)
        return [{'rewritten': self.tokenizer.decode(row, skip_special_tokens=True),
                 'factual_checks': factual_checks(original_bullet, self.tokenizer.decode(row, skip_special_tokens=True)),
                 'eligible_for_automatic_apply': False} for row in outputs]

    def load_model(self, model_path):
        loaded = T5ResumeRewriter(model_path=model_path)
        self.model, self.tokenizer, self.trained = loaded.model, loaded.tokenizer, loaded.trained


def create_training_data_from_sources(path='data/t5_training_data.csv'):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Provide licensed, reviewed rewriting pairs at {path}.')
    data = pd.read_csv(path, keep_default_na=False)
    required = {'source_id', 'original', 'improved', 'jd_context'}
    if not required.issubset(data.columns) or any(data[key].astype(str).str.strip().eq('').any() for key in ('source_id','original','improved')):
        raise ValueError('CSV requires source_id, original, improved, jd_context; original and improved must be nonempty.')
    return data


if __name__ == '__main__':
    raise SystemExit('Use python -m training.train_rewriter with --data and --provenance. No untrained demo outputs.')
