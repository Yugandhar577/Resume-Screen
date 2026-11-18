import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import main
from training.validate_reviewed_data import validate_manifest
from ml.local_rewriter import generate_proposals


class ReviewPolicyTests(unittest.TestCase):
    def test_ai_approval_requires_explicit_synthetic_mode_and_never_attests_human_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            data={'task':'job_bias','source_url':'urn:fictional:test','license':'Fixture only',
                  'label_method':'AI review','reviewer':'AI fixture','review_type':'ai','reviewed':False,
                  'ai_reviewed':True,'synthetic':True}
            path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(ValueError):validate_manifest(path,'job_bias')
            self.assertFalse(validate_manifest(path,'job_bias',allow_ai_reviewed_synthetic=True)['reviewed'])
            for changed in ({'reviewed':True},{'synthetic':False},{'ai_reviewed':False}):
                path.write_text(json.dumps({**data,**changed}),encoding='utf-8')
                with self.assertRaises(ValueError):validate_manifest(path,'job_bias',allow_ai_reviewed_synthetic=True)

    def test_t5_cannot_escape_shared_fact_checks(self):
        source='- Helped with Python reports for 5 teams.'
        invented={'edits':[{'line':1,'original':source,'proposed':'- Led Python reports for 50 teams.','reason':'Unsafe fixture'}]}
        with patch('ml.research_inference.t5_candidates',return_value=invented), patch('ml.local_rewriter.local_model.generate') as model:
            result=generate_proposals(source,'Data Analyst',engine='t5')
        self.assertEqual(result['rejected'],1)
        self.assertFalse(any(row['id'].startswith('local-') for row in result['proposals']))
        model.assert_not_called()

    def test_research_is_opt_in_and_does_not_change_coverage(self):
        client=TestClient(main.app)
        with patch('ml.research_inference.jd_experiment',return_value={'available':True,'labels':[]}) as experiment:
            payload={'resume':'Python','jobDescription':'Python SQL'}
            baseline=client.post('/api/analyze',json=payload).json()
            experiment.assert_not_called()
            research=client.post('/api/analyze',json={**payload,'useResearch':True}).json()
            self.assertTrue(research['detailed_results']['jd_experiment']['available'])
            self.assertEqual(research['analysis']['atsScore'],baseline['analysis']['atsScore'])
        self.assertEqual(client.post('/api/rewrite-proposals',json={**payload,'engine':'cloud'}).status_code,422)

    def test_career_requires_an_explicit_catalog_role(self):
        client=TestClient(main.app)
        self.assertEqual(client.get('/api/career-roles?query=x').status_code,422)
        with patch('ml.career_experiment.explore',side_effect=ValueError('Select a role')):
            self.assertEqual(client.get('/api/career-explore?code=unknown').status_code,422)


class GraphAndResourceTests(unittest.TestCase):
    def test_entity_graph_preserves_exact_spans_and_observed_containment(self):
        from ml.entity_graph import EntityGraph
        graph=EntityGraph.__new__(EntityGraph)
        class EmptyDoc:ents=[]
        graph.nlp=lambda text:EmptyDoc()
        text='  Software Engineer\nNo Python experience.\nSkills: SQL\n'
        nodes,relations=graph.nodes(text)
        self.assertTrue(any(n['kind']=='SKILL' and n['name']=='Python' for n in nodes))
        for node in nodes:self.assertEqual(text[node['start']:node['end']],node['quote'])
        for relation in relations:
            first,second=nodes[relation['source']],nodes[relation['target']]
            if relation['kind']=='source_contains_entity':
                self.assertIn(second['quote'],first['quote'])
        self.assertTrue(any('No Python' in n['quote'] for n in nodes))

    def test_weak_label_all_abstain_is_not_negative(self):
        import numpy as np
        from ml.weak_wording import posterior,votes
        self.assertEqual(votes('Hello there','Biased'),[-1,-1,-1])
        self.assertIsNone(posterior(np.ones((3,3,2))*.5,np.array([.5,.5]),[-1,-1,-1]))

    @unittest.skipUnless(os.name=='nt','Windows Job Object regression')
    def test_windows_parser_memory_limit_is_enforced(self):
        code='from ml.windows_limits import limit_current_process\nlimit_current_process(64*1024**2)\ntry:\n x=bytearray(128*1024**2)\nexcept MemoryError:\n print("limited")\n'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('limited',result.stdout)


if __name__=='__main__':unittest.main()
