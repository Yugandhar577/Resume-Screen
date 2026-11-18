"""Fail closed on absent provenance or unreviewed rewriting/bias examples."""
import json
import hashlib
from pathlib import Path


def review_permitted(data, allow_ai_reviewed_synthetic=False):
    if data.get('review_type') == 'ai':
        return (allow_ai_reviewed_synthetic and data.get('synthetic') is True
                and data.get('ai_reviewed') is True and data.get('reviewed') is False)
    return data.get('review_type') in (None, 'human') and data.get('reviewed') is True


def validate_manifest(path, task, dataset_path=None, allow_ai_reviewed_synthetic=False):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'source_url', 'license', 'label_method', 'reviewer', 'reviewed', 'task'}
    if not required.issubset(data) or data['task'] != task:
        raise ValueError('Dataset provenance is incomplete or belongs to another task.')
    if not review_permitted(data, allow_ai_reviewed_synthetic) or not data['reviewer'].strip():
        raise ValueError('This training path requires documented factual/human review. Synthetic teacher outputs are not automatically reviewed.')
    if not data['license'].strip() or not data['source_url'].strip():
        raise ValueError('Source and usage license are required.')
    if dataset_path and hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest() != data.get('data_sha256'):
        raise ValueError('The reviewed dataset checksum does not match its provenance manifest.')
    return data
