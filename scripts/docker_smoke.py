"""Check the running container stack through the frontend proxy; no external AI calls."""
import argparse
import os
from pathlib import Path
import requests
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:3000')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env')
    session = requests.Session()
    token = os.getenv('API_AUTH_TOKEN', '').strip()
    if token:
        session.headers['Authorization'] = 'Bearer ' + token
    url = args.url.rstrip('/')
    response = session.get(url, timeout=15); response.raise_for_status()
    health = session.get(url + '/health', timeout=15); health.raise_for_status()
    status = health.json()
    assert status['status'] == 'healthy'
    payload = {'resume': 'Alex Example\nSkills\nPython SQL\nExperience\n- Worked on Python services for 5 users',
               'jobDescription': 'Software Engineer\nRequired: Python SQL', 'useAI': False}
    response = session.post(url + '/api/analyze', json=payload, timeout=30); response.raise_for_status()
    result = response.json()
    assert result['analysis']['atsScore'] == 100
    proposals = result['optimization']['proposals']
    assert proposals
    revised = session.post(url + '/api/optimize-preview', json={**payload, 'acceptedEdits': [proposals[0]['id']], 'template':'engineering'}, timeout=30)
    revised.raise_for_status()
    assert 'Contributed to Python services for 5 users' in revised.json()['optimizedResume']
    assert revised.json()['comparison']['delta'] == 0
    font = session.get(url + '/fonts/NotoSans-Regular.ttf', timeout=15); font.raise_for_status()
    assert len(font.content) > 100000
    fixture = root / 'tmp/pdfs/export-regression.pdf'
    if fixture.is_file():
        with fixture.open('rb') as stream:
            response = session.post(url + '/api/analyze-pdf', files={'file':('resume.pdf', stream, 'application/pdf')},
                                    data={'jobDescription':'Python SQL', 'useAI':'false'}, timeout=30)
        response.raise_for_status()
        assert response.json()['pdf_processed']
        assert 'FINAL LINE' in response.json()['optimization']['sourceResume']
        print('PDF upload and extraction through frontend proxy: passed')
    print('Frontend, health proxy, analysis, accepted edits, zero-gain comparison, PDF font: passed')
    print(f"Container capabilities: OCR={status['capabilities']['ocr']}; experimental matching={status['capabilities']['trained_matching']}")


if __name__ == '__main__':
    main()
