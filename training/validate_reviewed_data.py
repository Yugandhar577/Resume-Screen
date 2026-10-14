"""Fail closed on absent provenance or unreviewed rewriting/bias examples."""
import json
import hashlib
from pathlib import Path


def validate_manifest(path, task, dataset_path=None):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'source_url', 'license', 'label_method', 'reviewer', 'reviewed', 'task'}
    if not required.issubset(data) or data['task'] != task:
        raise ValueError('Dataset provenance is incomplete or belongs to another task.')
    if data['reviewed'] is not True or not data['reviewer'].strip():
        raise ValueError('This training path requires documented factual/human review. Synthetic teacher outputs are not automatically reviewed.')
    if not data['license'].strip() or not data['source_url'].strip():
        raise ValueError('Source and usage license are required.')
    if dataset_path and hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest() != data.get('data_sha256'):
        raise ValueError('The reviewed dataset checksum does not match its provenance manifest.')
    return data
