"""Capability reporting for model artifacts; no placeholders presented as loaded AI."""
import hashlib
import json
from pathlib import Path


def artifact_status(directory, task):
    root = Path(directory)
    manifest = root / 'manifest.json'
    if not manifest.is_file():
        return {'available': False, 'reason': f'No evaluated {task} artifact is installed.'}
    try:
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        if metadata.get('task') != task or not metadata.get('training_completed') or not metadata.get('evaluation', {}).get('test'):
            raise ValueError('Training, task, and held-out evaluation provenance are required.')
        checkpoint = root / metadata.get('checkpoint', 'model.pt')
        if checkpoint.parent.resolve() != root.resolve():
            raise ValueError('Checkpoint path must stay in its model directory.')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata.get('checkpoint_sha256'):
            raise ValueError('Checkpoint checksum mismatch.')
        return {'available': bool(metadata.get('deployment_ready', False)), 'trained': True,
                'reason': metadata.get('deployment_reason', 'Inference integration and deployment validation are required.'),
                'version': metadata['version'], 'provenance': metadata}
    except Exception as exc:
        return {'available': False, 'reason': str(exc)}
