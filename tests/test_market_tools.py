import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import main
from ml.salary import compare,extract_offers
from ml.career_matching import evidence_for
from ml.skills import skill_evidence


class SalaryTests(unittest.TestCase):
    def test_indian_units_and_monthly_ranges(self):
        for text,low,high in [('INR 8-12 LPA CTC',800000,1200000),('Rs. 50,000 - Rs. 70,000 per month base salary',600000,840000),('INR 1 crore annually',10000000,10000000),('INR 1,20,000 per month',1440000,1440000)]:
            offer=extract_offers(text)[0]
            self.assertTrue(offer['valid']);self.assertEqual((offer['annual_low'],offer['annual_high']),(low,high))
        self.assertEqual(extract_offers('Requires 5 years experience. Salary negotiable.'),[])

    def test_inconsistent_units_currencies_and_ranges_are_rejected(self):
        for text in ['INR 8 LPA per month','INR 8-$12 LPA CTC','INR 12-8 LPA','INR -5 LPA','Up to INR 8 LPA','INR 10 LPA+']:
            self.assertFalse(extract_offers(text)[0]['valid'])

    def test_basis_currency_experience_and_multiple_components_do_not_produce_mismatch(self):
        for text,years,basis,expected in [('USD 10000 annually CTC',2,'ctc','not_comparable'),('INR 5 LPA base',2,'base','not_comparable'),('INR 5 LPA',2,'unspecified','reference_only'),('INR 5 LPA CTC',8,'ctc','outside_experience_range'),('INR 8 LPA base plus INR 2 LPA bonus',2,'ctc','needs_offer')]:
            self.assertEqual(compare(text,'syncwell-5',years,pay_basis=basis)['comparison'],expected)
        self.assertEqual(compare('INR 8 LPA base','syncwell-5',2,pay_basis='ctc')['comparison'],'not_comparable')

    def test_actual_reference_comparison_and_source_date(self):
        for text,expected in [('INR 5 LPA CTC','below_reference'),('INR 8 LPA CTC','overlaps_reference'),('INR 12 LPA CTC','above_reference')]:
            result=compare(text,'syncwell-5',2)
            self.assertEqual(result['comparison'],expected)
            self.assertEqual(result['reference']['as_of'],'2025')
            self.assertIn('no verified city',result['location_note'])

    def test_api_validates_user_benchmark_and_does_not_fetch_its_url(self):
        client=TestClient(main.app)
        reference={'title':'Unit fixture, not market data','role':'Fixture role','url':'https://example.org/fixture','as_of':'2025',
          'pay_basis':'base','geography':'Fixture location','method':'Unit fixture','license':'Test fixture',
          'experience_min':0,'experience_max':3,'annual_low':100000,'annual_high':200000}
        body={'offerText':'INR 150000 annually base','experienceYears':2,'reference':reference}
        with patch('requests.get') as fetch:
            response=client.post('/api/salary-compare',json=body)
            self.assertEqual(response.status_code,200);fetch.assert_not_called()
            self.assertTrue(response.json()['reference']['user_provided'])
        for changed in ({'annual_high':1},{'url':'javascript:alert(1)'},{'as_of':'2099'},{'country':'USA'}):
            self.assertEqual(client.post('/api/salary-compare',json={**body,'reference':{**reference,**changed}}).status_code,422)


class CareerTests(unittest.TestCase):
    def test_direct_skill_evidence_respects_aliases_and_negation(self):
        resume='Skills: ReactJS, SQL\nNo Python experience.'
        evidence=skill_evidence(resume)
        self.assertTrue(evidence_for('React',resume,evidence))
        self.assertTrue(evidence_for('SQL',resume,evidence))
        self.assertFalse(evidence_for('Python (computer programming)',resume,evidence))
        self.assertFalse(evidence_for('medical diagnosis',resume,evidence))
        self.assertFalse(evidence_for('design SQL Server databases',resume,evidence))
        self.assertFalse(evidence_for('SQL Server',resume,evidence))

    def test_unknown_roles_and_unavailable_index_are_explicit(self):
        client=TestClient(main.app)
        with patch('ml.career_matching.recommend',side_effect=ValueError('Unknown target occupation')):
            self.assertEqual(client.post('/api/career-recommendations',json={'resume':'Python','targetRoleCodes':['unknown']}).status_code,422)
        with patch('ml.career_matching.recommend',side_effect=FileNotFoundError):
            self.assertEqual(client.post('/api/career-recommendations',json={'resume':'Python'}).status_code,503)

    def test_semantic_passage_matching_is_opt_in_and_cannot_change_baseline(self):
        client=TestClient(main.app)
        body={'resume':'Python','jobDescription':'Python SQL'}
        with patch('ml.evidence_matching.analyze',return_value={'available':True,'items':[]}) as matching:
            before=client.post('/api/analyze',json=body).json();matching.assert_not_called()
            after=client.post('/api/analyze',json={**body,'useEvidence':True}).json()
            self.assertEqual(before['analysis']['atsScore'],after['analysis']['atsScore'])
            self.assertTrue(after['detailed_results']['evidence_matching']['available'])


if __name__=='__main__':unittest.main()
