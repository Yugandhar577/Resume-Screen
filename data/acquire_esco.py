"""Explicit public-taxonomy download; no candidate content is transmitted."""
import hashlib
import json
from pathlib import Path
import requests


def main():
    root = Path('Datasets/esco'); root.mkdir(parents=True, exist_ok=True)
    url = 'https://ec.europa.eu/esco/api/search'
    parameters = {'text':'', 'type':'occupation', 'language':'en', 'limit':5000, 'selectedVersion':'v1.1.2'}
    response = requests.get(url, params=parameters, timeout=(15,90))
    response.raise_for_status()
    raw = response.content
    if len(raw) > 40*1024**2:
        raise ValueError('Taxonomy response too large.')
    data = response.json(); results = data.get('_embedded', {}).get('results', [])
    if len(results) != data.get('total'):
        raise ValueError('Incomplete occupation catalog. Stop rather than silently omit roles.')
    roles = [{'code':row['code'], 'title':row['preferredLabel']['en'], 'uri':row['uri']} for row in results if row.get('code') and row.get('preferredLabel',{}).get('en')]
    (root/'occupations.json').write_text(json.dumps(roles,ensure_ascii=False,indent=2),encoding='utf-8')
    provenance = {'source':response.url, 'version':'v1.1.2', 'attribution':'European Commission, ESCO classification',
                  'reuse_notice':'https://commission.europa.eu/legal-notice_en',
                  'usage':'Public occupational names/codes with attribution; taxonomy API software is not redistributed.',
                  'raw_sha256':hashlib.sha256(raw).hexdigest(), 'catalog_sha256':hashlib.sha256((root/'occupations.json').read_bytes()).hexdigest(),
                  'occupations':len(roles)}
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(f'Acquired {len(roles)} ESCO v1.1.2 occupations.')


if __name__ == '__main__': main()
