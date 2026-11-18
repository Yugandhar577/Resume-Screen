import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

import main
from integrations.local_llm import OllamaClient, LocalModelError
from ml.local_review import review_locally, recover_quote
from ml.local_rewriter import check_facts, generate_proposals, read_token, apply_local_edits, _sign, _hash
from ml.pdf_processing import extract_pdf_isolated
from data.review_queue import seed, export_reviewed, read_csv
from integrations.job_boards import RemoteOK
from training.validate_reviewed_data import validate_manifest


class TransportTests(unittest.TestCase):
    def test_compose_service_is_allowed_without_accepting_other_network_hosts(self):
        client = OllamaClient()
        with patch.dict(os.environ, {'OLLAMA_BASE_URL': 'http://ollama:11434'}):
            self.assertEqual(client._url('/api/chat'), 'http://ollama:11434/api/chat')
        for url in ('http://ollama', 'http://ollama:8080', 'http://ollama:invalid',
                    'http://ollama.example:11434', 'http://192.168.1.10:11434',
                    'http://user:password@ollama:11434', 'http://ollama:11434?redirect=evil',
                    'http://ollama:11434/extra', 'https://ollama:11434'):
            with self.subTest(url=url), patch.dict(os.environ, {'OLLAMA_BASE_URL': url}):
                with self.assertRaises(LocalModelError):
                    client._url('/api/chat')

    def test_remote_url_and_cloud_models_are_rejected(self):
        client = OllamaClient()
        for url in ('https://127.0.0.1:11434', 'http://evil.example', 'http://localhost@evil.example', 'http://127.0.0.1:11434/extra'):
            with patch.dict(os.environ, {'OLLAMA_BASE_URL': url}):
                with self.assertRaises(LocalModelError): client._url('/api/chat')
        with patch.dict(os.environ, {'OLLAMA_MODEL': 'gemma4:31b-cloud'}), patch.object(client, '_request') as request:
            self.assertFalse(client.status(refresh=True)['available']); request.assert_not_called()

    def test_cloud_alias_does_not_send_candidate_content(self):
        client = OllamaClient()
        with patch.object(client, '_request', return_value={'models': [{'name': client.model, 'remote_host': 'https://ollama.com'}]}) as request:
            with self.assertRaises(LocalModelError): client.generate('Review', {'resume':'private'}, {})
            self.assertEqual(request.call_count, 1)
            self.assertEqual(request.call_args.args[0], '/api/tags')

    def test_output_limit_and_invalid_json_are_explicit(self):
        client = OllamaClient()
        with patch.object(client, 'status', return_value={'available': True}):
            for result in ({'done_reason':'length'}, {'message':{'content':'not json'}}):
                with patch.object(client, '_request', return_value=result):
                    with self.assertRaises(LocalModelError): client.generate('Review', {}, {})
            with patch.object(client, '_request') as request:
                with self.assertRaises(LocalModelError): client.generate('Review', {'resume':'a'*15000}, {})
                request.assert_not_called()

    def test_concurrent_model_calls_do_not_fill_gpu_queue(self):
        client = OllamaClient()
        with patch.object(client, 'status', return_value={'available': True}):
            client._slot.acquire()
            try:
                with self.assertRaises(LocalModelError): client.generate('Review', {}, {})
            finally:
                client._slot.release()


class EvidenceTests(unittest.TestCase):
    def test_whitespace_only_quotations_recover_exact_source(self):
        self.assertEqual(recover_quote('Skills Python SQL', 'Skills\nPython SQL'), 'Skills\nPython SQL')
        self.assertIsNone(recover_quote('Skills Python AWS', 'Skills\nPython SQL'))

    def test_all_generated_evidence_is_source_checked(self):
        report = {'reviewers':[{'persona': name, 'rubric':'Evidence', 'strengths':[{'statement':'Claim', 'evidence':['Fabricated quote']}], 'concerns':[], 'recommendations':[]} for name in ('ATS & structure','Technical evidence','Recruiter clarity')],
                  'alignment':[{'job_quote':'SQL', 'resume_quote':'Invented', 'status':'evidenced', 'explanation':'No'}],
                  'job_flags':[{'phrase':'Invented', 'category':'vague requirements','explanation':'No','suggestion':'No'}],
                  'career_options':[{'role':'Director','evidence':['Python'],'existing_skills':['AWS'],'development':[],'explanation':'No'}],
                  'entities':[{'name':'Acme','kind':'company','source':'resume','quote':'Invented'}]}
        with patch('ml.local_review.local_model.generate', return_value=report), patch('ml.local_review.local_model.status', return_value={}):
            result = review_locally('Python', 'SQL')
        self.assertEqual(result['rejected_unverified_items'], 7)
        self.assertEqual(result['alignment'], [])
        self.assertEqual(result['career_options'], [])

    def test_baseline_score_explanation_is_an_actual_perturbation(self):
        result = main.careerlens_engine.analyze_resume_jd_pair('Python SQL', 'Python SQL Docker')
        for item in result['detailed_results']['score_explanation']['items']:
            revised = main.careerlens_engine.analyze_resume_jd_pair('SQL' if item['skill'] == 'Python' else 'Python', 'Python SQL Docker')
            self.assertEqual(item['score_without_skill'], revised['analysis']['atsScore'])


class ProposalTests(unittest.TestCase):
    source = '- Helped with Python APIs at Acme in 2023 serving 50 users, not production'
    proposed = '- Supported Python APIs at Acme in 2023 serving 50 users, not production'

    def test_adversarial_protected_facts(self):
        self.assertTrue(check_facts(self.source, self.proposed)['valid'])
        for changed in (self.proposed.replace('50','500'), self.proposed.replace('50 users','50 teams'), self.proposed.replace('50','-50'), self.proposed.replace('Python','AWS'), self.proposed.replace('Acme','Other'), self.proposed.replace('not ',''), self.proposed.replace('Supported','Led'), self.proposed+'\n- Added claims'):
            self.assertFalse(check_facts(self.source, changed)['valid'])

    def test_signed_edits_are_bound_to_resume_job_and_expiration(self):
        batch = {'edits':[{'line':1, 'original':self.source, 'proposed':self.proposed, 'reason':'Concise wording'}]}
        verdict = {'verdicts':[{'line':1,'equivalent':True,'explanation':'Same supporting role'}]}
        with patch('ml.local_rewriter.local_model.generate', side_effect=[batch, verdict]):
            result = generate_proposals(self.source, 'Python')
        token = result['proposalToken']; keys = [e['id'] for e in result['proposals']]
        self.assertEqual(apply_local_edits(self.source, 'Python', keys, 'original', token)['resume'], self.proposed)
        for altered, resume, job in ((token[:-1]+('0' if token[-1]!='0' else '1'), self.source,'Python'),(token,self.source+'!', 'Python'),(token,self.source,'SQL')):
            with self.assertRaises(ValueError): read_token(altered,resume,job)
        expired = _sign({'resume_sha256':_hash(self.source),'job_sha256':_hash('Python'),'expires':0,'edits':[]})
        with self.assertRaises(ValueError): read_token(expired,self.source,'Python')

    def test_semantic_rejection_prevents_local_proposal(self):
        freeform=self.proposed.replace('Python APIs','APIs using Python')
        batch = {'edits':[{'line':1, 'original':self.source, 'proposed':freeform, 'reason':'Edit'}]}
        verdict = {'verdicts':[{'line':1,'equivalent':False,'explanation':'Changed implication'}]}
        with patch('ml.local_rewriter.local_model.generate', side_effect=[batch,verdict]):
            result = generate_proposals(self.source,'Python')
        self.assertEqual(result['rejected'],1)
        self.assertFalse(any(e['id'].startswith('local-') for e in result['proposals']))

    def test_exact_rule_certificate_does_not_need_model_equivalence(self):
        batch={'edits':[{'line':1,'original':self.source,'proposed':self.proposed,'reason':'Concise wording'}]}
        with patch('ml.local_rewriter.local_model.generate',return_value=batch) as generator:
            result=generate_proposals(self.source,'Python')
        self.assertEqual(generator.call_count,1)
        self.assertEqual(result['generated_proposal_count'],1)
        self.assertIn('certificate',result['proposals'][0]['validation']['semantic_check'])

    def test_missing_terminal_fullstop_is_restored_only_for_exact_known_transform(self):
        original=self.source+'.'
        batch={'edits':[{'line':1,'original':original,'proposed':self.proposed,'reason':'Clarity'}]}
        with patch('ml.local_rewriter.local_model.generate',return_value=batch) as generator:
            result=generate_proposals(original,'Python')
        self.assertEqual(generator.call_count,1)
        self.assertEqual(result['proposals'][0]['proposed'],self.proposed+'.')
        self.assertIn('formatting_restoration',result['proposals'][0])
        batch['edits'][0]['proposed']=self.proposed.replace('50','500')
        with patch('ml.local_rewriter.local_model.generate',return_value=batch):
            result=generate_proposals(original,'Python')
        self.assertEqual(result['rejected'],1)

    def test_unavailable_rewriting_returns_service_error(self):
        with patch.object(main, 'generate_proposals', side_effect=LocalModelError('Local model unavailable')):
            response = TestClient(main.app).post('/api/rewrite-proposals',json={'resume':self.source,'jobDescription':'Python'})
        self.assertEqual(response.status_code,503)


class DocumentAndReviewTests(unittest.TestCase):
    def test_remoteok_skips_terms_and_retains_attribution(self):
        body = [{'legal':'Source terms'}, {'id':'fixture','position':'Python Engineer','description':'<p>SQL</p><script>bad()</script>','url':'https://remoteok.com/remote-jobs/fixture'},
                {'id':'unsafe','position':'Bad','url':'https://evil.example'}]
        response = MagicMock(); response.__enter__.return_value = response
        response.iter_content.return_value = [json.dumps(body).encode()]
        board = RemoteOK()
        with patch('integrations.job_boards.requests.get',return_value=response) as request:
            jobs = board.search('Python')['jobs']; board.search('SQL')
            self.assertEqual(request.call_count,1)
        self.assertEqual(len(jobs),1); self.assertEqual(jobs[0]['source'],'Remote OK')
        self.assertNotIn('bad()',jobs[0]['description'])

    def test_document_timeout_is_explicit_and_process_is_bounded(self):
        with patch('subprocess.run',side_effect=subprocess.TimeoutExpired('worker',45)) as run:
            with self.assertRaises(Exception) as error: extract_pdf_isolated(b'%PDF-')
            self.assertEqual(error.exception.status_code,408)
            self.assertEqual(run.call_args.kwargs['timeout'],45)

    def test_pending_drafts_cannot_be_exported_as_reviewed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source=root/'source.csv'; metadata=root/'source.json'; queue=root/'queue.csv'; output=root/'reviewed.csv'
            source.write_text('source_id,original,improved,jd_context\nsynthetic-1,- Helped with Python,- Supported Python,Python\n',encoding='utf-8')
            metadata.write_text(json.dumps({'source_url':'urn:fictional:test','license':'Test fixture only'}),encoding='utf-8')
            seed(source,metadata,queue,'resume_rewriting')
            with self.assertRaises(ValueError): export_reviewed(queue,output)
            rows=read_csv(queue); rows[0].update(review_status='approved',reviewer='Unit fixture reviewer')
            with queue.open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            self.assertEqual(export_reviewed(queue,output),1)
            validate_manifest(output.with_suffix('.provenance.json'),'resume_rewriting',output)
            output.write_text('tampered',encoding='utf-8')
            with self.assertRaises(ValueError): validate_manifest(output.with_suffix('.provenance.json'),'resume_rewriting',output)


if __name__ == '__main__':
    unittest.main()
