import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, DictionaryObject, DecodedStreamObject
import main
from ml.requirements import requirements_check
from ml.rewrite_workflow import proposals, apply_edits, validate_rewrite, reorder_sections
from ml.skills import extract_skills
from ml.pdf_processing import extract_pdf, ocr_available
from ml.hiring_simulator import HiringSimulator
from ml.local_review import review_locally
from ml.semantic_matching import SemanticMatcher
from ml.model_registry import artifact_status
from training.validate_reviewed_data import validate_manifest
from integrations.job_boards import Arbeitnow, plain_text


SOURCE = 'Alex Doe\nalex@example.com\nExperience\nEngineer @ Acme (2021-2023)\n- Worked on Python APIs serving 50 users, not production\n- Helped with SQL migration\nEducation\nBSc Computer Science\nSkills\nPython SQL'


class RewriteTests(unittest.TestCase):
    def test_accept_reject_and_zero_gain(self):
        client = TestClient(main.app)
        original = client.post('/api/analyze', json={'resume': SOURCE, 'jobDescription': 'Required: Python SQL AWS'}).json()
        edits = original['optimization']['proposals']
        self.assertEqual(len(edits), 2)
        response = client.post('/api/optimize-preview', json={'resume': SOURCE, 'jobDescription': 'Required: Python SQL AWS', 'acceptedEdits': [edits[0]['id']], 'template': 'engineering'})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn('Contributed to Python APIs serving 50 users, not production', result['optimizedResume'])
        self.assertIn('Helped with SQL migration', result['optimizedResume'])
        self.assertNotIn('AWS', result['optimizedResume'])
        self.assertEqual(result['comparison']['delta'], 0)
        self.assertEqual(result['optimization']['sourceResume'], SOURCE)
        self.assertCountEqual(result['detailed_results']['skill_evidence']['SQL'], ['- Helped with SQL migration', 'Python SQL'])
        self.assertEqual(apply_edits(SOURCE, [])['resume'], SOURCE)

    def test_stale_unknown_and_duplicate_ids(self):
        key = proposals(SOURCE)[0]['id']
        for ids, text in [([key, key], SOURCE), (['fake'], SOURCE), ([key], SOURCE.replace('50', '500'))]:
            with self.assertRaises(ValueError):
                apply_edits(text, ids)
        with self.assertRaises(ValueError):
            apply_edits(SOURCE, [], 'invented')

    def test_adversarial_facts_cannot_be_changed(self):
        facts = ['Python at Acme in 2021 serving 50 users', 'SQL, not AWS', 'migration failed; helped, not led', 'supported 5 engineers; did not manage them', 'only basic Python experience', '50% latency decrease in 2022']
        for fact in facts:
            source = '- Worked on ' + fact
            proposed = '- Contributed to ' + fact
            self.assertTrue(validate_rewrite(source, proposed)['valid'])
            for malicious in [proposed + ' and AWS', proposed.replace('Contributed to', 'Led'), proposed.replace(fact, fact.upper()), proposed.replace(fact, 'achieved 10 million revenue')]:
                self.assertFalse(validate_rewrite(source, malicious)['valid'])

    def test_templates_preserve_all_content(self):
        for template in ('engineering', 'data', 'general'):
            reordered = reorder_sections(SOURCE, template)
            from collections import Counter
            self.assertEqual(Counter(SOURCE.split()), Counter(reordered.split()))
            self.assertIn('Engineer @ Acme (2021-2023)', reordered)
            self.assertTrue(reordered.startswith('Alex Doe\nalex@example.com'))

    def test_requirements_negation_depth_and_priority(self):
        resume = 'No AWS experience\nBasic Python\nSQL\nBSc Computer Science\n3 years of software experience'
        jd = 'Required\nExpert Python\nAWS\nPreferred\nSQL\n5 years of software experience\nPhD\nJava not required'
        checks = requirements_check(resume, jd)['items']
        mapped = {item['requirement']: item for item in checks}
        self.assertEqual(mapped['Python']['status'], 'unverified')
        self.assertEqual(mapped['AWS']['status'], 'not_evidenced')
        self.assertEqual(mapped['SQL']['priority'], 'preferred')
        self.assertEqual(mapped['Doctorate']['priority'], 'preferred')
        self.assertNotIn('Java', mapped)
        years = next(item for item in checks if item['kind'] == 'experience')
        self.assertEqual(years['status'], 'unverified')
        self.assertEqual(years['priority'], 'preferred')
        self.assertEqual(extract_skills('not only Python but SQL'), ['Python', 'SQL'])
        self.assertEqual(extract_skills('No Python experience, but SQL'), ['SQL'])
        inline = requirements_check('Python SQL', 'Required: Python; preferred: SQL')['items']
        self.assertEqual([item['priority'] for item in inline], ['required', 'preferred'])
        self.assertFalse(any(item['kind'] == 'education' for item in requirements_check('Python', 'Develop websites')['items']))

    def test_reviewers_have_distinct_rubrics_and_exact_evidence(self):
        result = main.careerlens_engine.analyze_resume_jd_pair(SOURCE, 'Python SQL')
        reviewers = result['detailed_results']['reviewers']
        self.assertEqual(len(set(item['rubric'] for item in reviewers)), 3)
        for reviewer in reviewers:
            self.assertEqual(reviewer['outcome'], 'review_needed')
            for finding in reviewer['strengths'] + reviewer['concerns']:
                for quote in finding['evidence']:
                    self.assertIn(quote, SOURCE)


class ProviderAndDataTests(unittest.TestCase):
    def test_semantic_declines_unsupported_occupation(self):
        matcher = SemanticMatcher.__new__(SemanticMatcher)
        matcher.metadata = {'supported_role_families':['Software Engineer']}
        with patch.object(matcher, 'status', return_value={'available':True}):
            result = matcher.analyze('Patient care experience', 'Nurse needed for patient care')
        self.assertFalse(result['available'])
        self.assertIn('supported', result['reason'])

    def test_private_auth_and_request_body_limit(self):
        import os
        client = TestClient(main.app)
        token = 'unit-test-token-which-is-long-enough'
        with patch.dict(os.environ, {'API_AUTH_TOKEN': token}):
            self.assertEqual(client.post('/api/analyze', json={'resume':'Python','jobDescription':'SQL'}).status_code, 401)
            response = client.post('/api/analyze', headers={'Authorization':'Bearer '+token}, json={'resume':'Python','jobDescription':'SQL'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['cache-control'], 'no-store')
            self.assertEqual(client.post('/api/analyze', headers={'Authorization':'Bearer '+token, 'Content-Length':str(7*1024*1024)}, content=b'{}').status_code, 413)

    def test_fitted_label_aggregation_keeps_abstentions(self):
        import importlib.util
        if not importlib.util.find_spec('snorkel'):
            self.skipTest('Optional Snorkel is not installed')
        import numpy as np
        from data.label_aggregation import SnorkelAggregator
        # Synthetic unit fixture only, never saved or represented as reviewed training data.
        votes = np.array([[1,1,1], [0,0,0], [1,-1,1], [0,0,-1]] * 10)
        fitted = SnorkelAggregator().fit(votes, ['a','b','c'], epochs=100)
        probabilities = fitted.predict([[1,1,1], [0,0,0], [-1,-1,-1]])
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertIsNone(probabilities[2])

    def review_payload(self):
        return {'atsFeedback': 'Review', 'techFeedback': 'Review', 'hrFeedback': 'Review',
                'reviewers': [{'persona': name, 'rubric': 'Check supplied evidence', 'strengths': [{'statement': 'Skill mentioned', 'evidence': ['Python']}], 'concerns': [], 'recommendations': []} for name in ['ATS & structure', 'Technical evidence', 'Recruiter clarity']]}

    def test_ai_rejects_fabricated_quotations(self):
        for fabricated in (False, True):
            payload = self.review_payload()
            if fabricated:
                payload['reviewers'][0]['strengths'][0]['evidence'] = ['Led 500 people']
            payload = {key: payload[key] for key in ('reviewers',)}
            payload.update(alignment=[], job_flags=[], career_options=[], entities=[])
            with patch('ml.local_review.local_model.generate', return_value=payload), patch('ml.local_review.local_model.status', return_value={}):
                result = review_locally('Python', 'SQL')
                self.assertEqual(len(result['reviewers']), 3)
                self.assertEqual(result['rejected_unverified_items'], 1 if fabricated else 0)
                self.assertEqual(len(result['reviewers'][0]['strengths']), 0 if fabricated else 1)

    def test_job_board_cache_attribution_and_safe_links(self):
        fixture = {'data': [{'slug': 'python-role', 'title': 'Python Engineer', 'company_name': 'Example', 'description': '<p>Python SQL</p><script>bad()</script>', 'url': 'https://www.arbeitnow.com/jobs/role', 'location': 'Berlin'}, {'slug': 'bad', 'url': 'https://evil.example/link'}]}
        response = MagicMock(); response.__enter__.return_value = response
        response.iter_content.return_value = [json.dumps(fixture).encode()]
        board = Arbeitnow()
        with patch('integrations.job_boards.requests.get', return_value=response) as fetch:
            result = board.search('python')
            board.search('sql')
            self.assertEqual(fetch.call_count, 1)
        self.assertEqual(len(result['jobs']), 1)
        self.assertEqual(result['jobs'][0]['source'], 'Arbeitnow')
        self.assertNotIn('bad()', result['jobs'][0]['description'])
        self.assertEqual(plain_text('<p>A &amp; B</p>'), 'A & B')

    def test_missing_or_unreviewed_models_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(SemanticMatcher(directory).status()['available'])
            self.assertFalse(artifact_status(directory, 'resume_rewriting')['available'])
            path = Path(directory) / 'provenance.json'
            path.write_text(json.dumps({'source_url': 'https://example.org', 'license': 'MIT', 'label_method': 'teacher', 'reviewer': '', 'reviewed': False, 'task': 'resume_rewriting'}))
            with self.assertRaises(ValueError): validate_manifest(path, 'resume_rewriting')

    def test_research_artifact_does_not_enable_deployment(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); content = b'unit-fixture-not-an-inference-model'
            (root/'model.pt').write_bytes(content)
            metadata = {'task':'career_trajectory','training_completed':True,'evaluation':{'test':{'auc':0.5}},
                        'checkpoint':'model.pt','checkpoint_sha256':hashlib.sha256(content).hexdigest(),
                        'version':'unit-fixture','deployment_ready':False,'deployment_reason':'Fairness evaluation missing'}
            (root/'manifest.json').write_text(json.dumps(metadata))
            status = artifact_status(root, 'career_trajectory')
            self.assertTrue(status['trained']); self.assertFalse(status['available'])
            (root/'model.pt').write_bytes(b'tampered')
            self.assertFalse(artifact_status(root, 'career_trajectory')['available'])


class PdfWorkflowTests(unittest.TestCase):
    def test_actual_two_column_order(self):
        writer = PdfWriter(); writer.add_blank_page(width=612, height=792); page = writer.pages[0]
        font = DictionaryObject({NameObject('/Type'):NameObject('/Font'), NameObject('/Subtype'):NameObject('/Type1'), NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        lines = ['BT /F1 12 Tf 40 750 Td (Alex Doe) Tj ET']
        for index in range(5):
            y = 700 - index * 25
            lines.extend([f'BT /F1 12 Tf 40 {y} Td (Left{index} Python) Tj ET', f'BT /F1 12 Tf 360 {y} Td (Right{index} SQL) Tj ET'])
        stream = DecodedStreamObject(); stream.set_data('\n'.join(lines).encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
        buffer = io.BytesIO(); writer.write(buffer)
        result = extract_pdf(buffer.getvalue())
        self.assertEqual(result['methods'], ['two_column_heuristic'])
        self.assertLess(result['text'].index('Left4'), result['text'].index('Right0'))
        self.assertIn('Alex Doe', result['text'])

    def test_correct_password_and_partial_scan_warning(self):
        from test_analysis import pdf_bytes
        self.assertIn('Python', extract_pdf(pdf_bytes('Python SQL', password='secret'), password='secret')['text'])
        result = extract_pdf(pdf_bytes('Python SQL', pages=2))
        self.assertTrue(any('Some pages' in warning for warning in result['warnings']))

    @unittest.skipUnless(ocr_available(), 'Optional local OCR is not installed')
    def test_real_scanned_pdf_ocr(self):
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new('RGB', (600, 800), 'white')
        font = ImageFont.truetype('public/fonts/NotoSans-Regular.ttf', 28)
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(['Alex Doe', 'Skills: Python SQL', 'Built 5 services']):
            draw.text((40, 40+index*70), line, font=font, fill='black')
        buffer = io.BytesIO(); image.save(buffer, format='PDF')
        result = extract_pdf(buffer.getvalue(), use_ocr=True)
        self.assertEqual(result['methods'], ['local_ocr'])
        self.assertIn('Python', result['text']); self.assertIn('SQL', result['text'])
        self.assertTrue(result['warnings'])


if __name__ == '__main__':
    unittest.main()
