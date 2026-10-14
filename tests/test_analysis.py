import io
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PyPDF2 import PdfWriter
from PyPDF2.generic import NameObject, DictionaryObject, DecodedStreamObject

import main
from ml.careerlens_engine import CareerLensEngine
from ml.resume_optimizer import ResumeOptimizer
from ml.skills import extract_skills
from data.synthetic_negatives import SyntheticNegativeGenerator


def pdf_bytes(text=None, password=None, pages=1):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    if text:
        page = writer.pages[0]
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 100 700 Td ({text}) Tj ET'.encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(stream)
    if password:
        writer.encrypt(password)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'GROQ_API_KEY': ''})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.client = TestClient(main.app)
        self.engine = CareerLensEngine()

    def analyze(self, resume='Python SQL', jd='Python SQL', **extra):
        return self.client.post('/api/analyze', json={'resume': resume, 'jobDescription': jd, **extra})

    def pdf(self, content, filename='resume.pdf', jd='Python SQL'):
        return self.client.post('/api/analyze-pdf', files={'file': (filename, content, 'application/pdf')}, data={'jobDescription': jd})

    def test_case_and_aliases(self):
        result = self.analyze('python POSTGRES K8s', 'Python PostgreSQL Kubernetes').json()
        self.assertEqual(result['analysis']['atsScore'], 100)
        self.assertEqual(result['analysis']['skillGaps'], [])

    def test_extra_skills_do_not_lower_coverage(self):
        self.assertEqual(self.analyze('Python SQL Docker').json()['analysis']['atsScore'], 100)

    def test_partial_and_unrelated_matches(self):
        self.assertEqual(self.analyze('Python', 'Python SQL').json()['analysis']['atsScore'], 50)
        self.assertEqual(self.analyze('Python', 'SQL').json()['analysis']['atsScore'], 0)

    def test_unsupported_job_has_no_fabricated_score(self):
        result = self.analyze('Python', 'We need a specialist in an unlisted discipline').json()
        self.assertIsNone(result['analysis']['atsScore'])
        self.assertIsNone(result['detailed_results']['career_trajectory'])

    def test_extended_vocabulary_and_boundaries(self):
        self.assertEqual(extract_skills('C++ C# Go Rust FastAPI Spark Excel'), ['C#', 'C++', 'Excel', 'FastAPI', 'Go', 'Rust', 'Spark'])
        self.assertEqual(extract_skills('go forward with cargo and JavaScript'), ['JavaScript'])

    def test_all_skills_retained_and_stable(self):
        text = 'Python SQL Docker AWS Kubernetes Java React Git Redis Flask Django Rust C++ C#'
        skills = extract_skills(text)
        self.assertGreater(len(skills), 10)
        self.assertEqual(skills, extract_skills(text))

    def test_graphs_are_input_specific(self):
        first = self.analyze('Python SQL', 'SQL').json()['detailed_results']['graph_analysis']
        second = self.analyze('React JavaScript', 'React').json()['detailed_results']['graph_analysis']
        self.assertNotEqual(first, second)
        self.assertEqual(first['resume_graph']['edges'][0]['source'], 'Python')
        self.assertEqual(self.client.get('/api/graph-data/latest').status_code, 410)

    def test_company_graph_and_evidence(self):
        result = self.analyze('Software Engineer @ Acme Corp (2024)\nBuilt Python services', 'Python').json()
        graph = result['detailed_results']['graph_analysis']['resume_graph']
        self.assertIn({'id': 'Acme Corp', 'type': 'company', 'weight': 1}, graph['nodes'])
        self.assertEqual(result['detailed_results']['skill_evidence']['Python'], ['Built Python services'])

    def test_blank_and_large_inputs_rejected(self):
        for text in ['', '   ', 'a' * 50001]:
            self.assertEqual(self.analyze(resume=text).status_code, 422)
        self.assertEqual(self.analyze(jd=' ').status_code, 422)

    def test_formatting_preserves_facts(self):
        original = '  Alex Doe\n\n\n• Built 5 Python services at Acme\n  alex@example.com  '
        result = self.analyze(original).json()
        self.assertIn('- Built 5 Python services at Acme', result['optimizedResume'])
        self.assertNotIn('CAREERLENS', result['optimizedResume'])
        self.assertTrue(result['optimization']['validation']['valid'])

    def test_validator_rejects_fabrication_and_removal(self):
        validator = ResumeOptimizer()
        self.assertFalse(validator.validate_optimization('Built 5 services', 'Built 50 services')['valid'])
        self.assertFalse(validator.validate_optimization('Python SQL', 'Python')['valid'])
        self.assertFalse(validator.validate_optimization('Python', 'Python AWS')['valid'])

    def test_bias_evidence_and_no_team_player_false_flag(self):
        self.assertFalse(self.engine._detect_bias('team player')['bias_detected'])
        result = self.engine._detect_bias('Seeking young and energetic candidates')
        self.assertEqual(result['flags'][0]['phrase'], 'young and energetic')
        self.assertNotIn('quality_score', result)

    def test_pdf_and_text_share_job_mapping(self):
        response = self.pdf(pdf_bytes('Python SQL'), filename='RESUME.PDF')
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['analysis']['atsScore'], 100)
        self.assertTrue(result['pdf_processed'])
        self.assertEqual(result['detailed_results']['matching'], self.analyze().json()['detailed_results']['matching'])

    def test_bad_empty_encrypted_and_oversized_pdfs(self):
        for content, filename in [(b'not a pdf', 'resume.pdf'), (b'%PDF-1.4 invalid', 'resume.pdf'), (b'Python', 'resume.txt'), (pdf_bytes(), 'resume.pdf'), (pdf_bytes('Python', password='secret'), 'resume.pdf')]:
            self.assertEqual(self.pdf(content, filename).status_code, 400)
        self.assertEqual(self.pdf(b'%PDF-' + b'x' * main.MAX_PDF_BYTES).status_code, 413)
        self.assertEqual(self.pdf(pdf_bytes(pages=31)).status_code, 413)
        self.assertEqual(self.pdf(pdf_bytes('Python'), jd=' ').status_code, 422)

    def test_pdf_extraction_tries_secondary_parser(self):
        with patch('PyPDF2.PdfReader', side_effect=ValueError('primary parser failed')):
            self.assertIn('Python SQL', main.extract_text_from_pdf(pdf_bytes('Python SQL')))

    def test_health_does_not_claim_trained_models(self):
        status = self.client.get('/health').json()
        self.assertFalse(status['ml_engine'])
        self.assertFalse(status['groq_available'])
        self.assertFalse(status['capabilities']['generative_rewriting'])

    def test_ai_is_opt_in_and_cannot_overwrite_matching(self):
        with patch.object(main.HiringSimulator, 'is_configured', return_value=True), patch.object(main.HiringSimulator, 'review', return_value={'atsFeedback': 'ATS review', 'techFeedback': 'Technical review', 'hrFeedback': 'Recruiter review'}) as review:
            self.analyze()
            review.assert_not_called()
            result = self.analyze(useAI=True).json()
            self.assertEqual(result['analysis']['atsScore'], 100)
            self.assertEqual(result['review_method'], 'Groq')

    def test_ai_failure_returns_computed_results_with_warning(self):
        with patch.object(main.HiringSimulator, 'is_configured', return_value=True), patch.object(main.HiringSimulator, 'review', side_effect=ValueError('secret')):
            result = self.analyze(useAI=True).json()
            self.assertEqual(result['analysis']['atsScore'], 100)
            self.assertEqual(result['review_method'], 'Rule_Based')
            self.assertNotIn('secret', str(result))
            self.assertGreater(len(result['warnings']), 1)

    def test_internal_failure_is_not_success(self):
        with patch.object(main.careerlens_engine, 'analyze_resume_jd_pair', side_effect=RuntimeError('private')):
            response = self.analyze()
            self.assertEqual(response.status_code, 500)
            self.assertNotIn('private', response.text)

    def test_negatives_remove_actual_aliases(self):
        generator = SyntheticNegativeGenerator()
        modified = generator.generate_hard_negative('python POSTGRES K8s', ['Python', 'PostgreSQL', 'Kubernetes'])
        self.assertEqual(extract_skills(modified), [])
        self.assertEqual(generator.create_contrastive_pairs('No listed skills', 'Python', ['Python']), [('No listed skills', 'Python', 1)])


if __name__ == '__main__':
    unittest.main()
