"""Distinct reviewer rubrics backed by exact source lines."""
import re


def evidence_assessments(resume, checks):
    lines = [line.strip() for line in resume.splitlines() if line.strip()]
    sections = [line for line in lines if re.fullmatch(r'(?:professional |work )?experience:?|skills:?|education:?|projects:?|summary:?', line, re.I)]
    quantified = [line for line in lines if re.search(r'\d+(?:\.\d+)?\s*(?:%|users|customers|requests|projects)', line, re.I)]
    bullets = [line for line in lines if re.match(r'^[-•*]\s', line)]
    matched = [item for item in checks['items'] if item['kind'] == 'skill' and item['evidence']]
    gaps = [item['requirement'] for item in checks['items'] if item['priority'] == 'required' and item['status'] == 'not_evidenced']
    contact = [line for line in lines if re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', line)]
    def finding(statement, quotes, recommendation=None):
        return {'statement': statement, 'evidence': quotes[:6], 'recommendation': recommendation}
    return [
        {'persona': 'ATS & structure', 'method': 'Rubric v1', 'outcome': 'review_needed',
         'strengths': [finding(f'{len(sections)} standard section heading(s) found.', sections)],
         'concerns': [finding('Required skills without evidence: ' + ', '.join(gaps), [], 'Confirm actual experience before adding these skills.')] if gaps else [],
         'recommendations': ['Use standard headings and readable text.', 'Check all qualifications against the evidence checklist.'],
         'rubric': 'Checks parseable text, recognized headings, and explicitly required skill evidence; never predicts acceptance.'},
        {'persona': 'Technical evidence', 'method': 'Rubric v1', 'outcome': 'review_needed',
         'strengths': [finding(f'{len(matched)} job skill(s) have resume evidence.', [quote for item in matched for quote in item['evidence']]),
                       finding(f'{len(quantified)} line(s) describe measurable impact.', quantified)],
         'concerns': [finding('A skill mention alone cannot establish proficiency or ownership.', [])],
         'recommendations': ['Explain your contribution, context, and impact using facts you can substantiate.'],
         'rubric': 'Checks demonstrated technologies and stated project impact; does not infer proficiency from keywords.'},
        {'persona': 'Recruiter clarity', 'method': 'Rubric v1', 'outcome': 'review_needed',
         'strengths': [finding('Contact information is present.' if contact else 'No email contact was recognized.', contact),
                       finding(f'{len(bullets)} experience/project bullet(s) found.', bullets)],
         'concerns': [finding('Missing standard section headings.', [], 'Add headings for existing content.')] if not sections else [],
         'recommendations': ['Keep identity, employers, role titles, dates, and achievements accurate.'],
         'rubric': 'Checks presentation and factual clarity; excludes protected traits, personality, and cultural fit.'},
    ]
