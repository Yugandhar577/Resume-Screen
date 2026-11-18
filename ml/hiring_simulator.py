"""Compatibility facade for local evidence review. No external provider support."""
from integrations.local_llm import local_model
from ml.local_review import review_locally

class HiringSimulator:
    @staticmethod
    def is_configured():
        return local_model.status()['available']

    def review(self, resume, job_description):
        report = review_locally(resume, job_description)
        fields = {'ATS & structure': 'atsFeedback', 'Technical evidence': 'techFeedback', 'Recruiter clarity': 'hrFeedback'}
        feedback = {'reviewers': report['reviewers']}
        for reviewer in report['reviewers']:
            feedback[fields[reviewer['persona']]] = ' '.join([f['statement'] for f in reviewer['strengths']+reviewer['concerns']]+reviewer['recommendations'])
        return feedback
