"""Explicit, pinned acquisition of the structured JobHop research release."""
import hashlib
import json
from pathlib import Path
import requests

REPO = 'aida-ugent/JobHop'
REVISION = '766921f12b66eb886841393ab0400321e0373cb4'


def main():
    root = Path('Datasets/jobhop')
    root.mkdir(parents=True, exist_ok=True)
    provenance = {'source': f'https://huggingface.co/datasets/{REPO}', 'revision': REVISION,
                  'license': 'CC BY 4.0',
                  'attribution': 'JobHop v2: Iman Johary, Guillaume Bied, Alexandru C. Mara, Tijl De Bie; Ghent University; data provided by VDAB.',
                  'labels': 'Pseudonymized observed careers, LLM-extracted ESCO assignments; not human-reviewed candidate recommendations.',
                  'restrictions': 'Research only here: no re-identification, protected-attribute inference, or consequential decisions. Fairness evaluation is required before deployment.',
                  'files': []}
    for filename in ['JobHop_v2_train.parquet', 'JobHop_v2_val.parquet', 'JobHop_v2_test.parquet', 'README.md']:
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{filename}'
        content = bytearray()
        with requests.get(url, timeout=(15, 60), stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > 20 * 1024 * 1024:
                    raise ValueError('Source exceeded the acquisition limit.')
        (root / filename).write_bytes(content)
        provenance['files'].append({'file': filename, 'url': url, 'bytes': len(content),
                                    'sha256': hashlib.sha256(content).hexdigest()})
        print(f'Downloaded {filename}: {len(content)} bytes', flush=True)
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
