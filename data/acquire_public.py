"""Download an explicitly licensed, revision-pinned synthetic matching corpus."""
import hashlib
import json
from pathlib import Path
import requests

REPO = 'michaelozon/candidate-matching-synthetic'
REVISION = '178ab864dcad9910c5670d43e4bdbbb901a11f18'
FILES = ['resumes/train-00000-of-00001.parquet', 'jobs/train-00000-of-00001.parquet',
         'matches/train-00000-of-00001.parquet', 'README.md']


def main():
    target = Path('Datasets/candidate-matching-synthetic')
    target.mkdir(parents=True, exist_ok=True)
    manifest = {'source': f'https://huggingface.co/datasets/{REPO}', 'revision': REVISION,
                'license': 'MIT (declared by dataset author)', 'author': 'Michael Ozon',
                'labels': 'Synthetic skills/rule-generated relevance, not human hiring decisions',
                'usage': 'Experimental training/evaluation only; do not claim real hiring validity.', 'files': []}
    for filename in FILES:
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{filename}'
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        if len(response.content) > 10 * 1024 * 1024:
            raise ValueError('Unexpectedly large source file.')
        path = target / filename.replace('/', '-')
        path.write_bytes(response.content)
        manifest['files'].append({'file': path.name, 'url': url,
                                  'sha256': hashlib.sha256(response.content).hexdigest(), 'bytes': len(response.content)})
        print(f'Downloaded {path.name}: {len(response.content)} bytes', flush=True)
    (target / 'provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Source provenance saved. Synthetic labels are explicitly identified.')


if __name__ == '__main__':
    main()
