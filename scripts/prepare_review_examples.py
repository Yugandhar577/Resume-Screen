"""Build fictional, unreviewed practice examples; never label these real hiring data."""
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.review_queue import seed


def main():
    root = Path(__file__).resolve().parent.parent
    directory = root/'data'/'review_examples'
    directory.mkdir(exist_ok=True)
    roles = [
        ('Software Engineer', 'Python APIs serving 50 test users', 'SQL reporting for 4 internal teams'),
        ('Data Analyst', 'Excel reports covering 12 stores', 'Python data checks for 3 source files'),
        ('Accountant', '40 weekly invoices using Excel', 'Bookkeeping for 2 fictional branches'),
        ('Teacher', 'Teaching activities for 25 students', 'lesson planning for 3 classes'),
        ('Nurse', 'Patient Care documentation for 8 patients under supervision', 'handover notes for 2 shifts'),
        ('Customer Support Specialist', 'Customer Service for 30 daily requests', 'support documentation for 5 common issues'),
        ('Project Coordinator', 'Project Management tracking for 4 workstreams', 'weekly updates for 3 teams'),
        ('Frontend Developer', 'React components for 6 internal screens', 'CSS fixes for 2 test browsers'),
        ('Operations Assistant', 'inventory counts across 7 storage areas', 'process documentation for 4 tasks'),
        ('Designer', 'Figma prototypes for 3 sample pages', 'UX research notes from 6 practice interviews'),
        ('Marketing Assistant', 'Excel campaign reports for 2 fictional products', 'content checks for 10 draft posts'),
        ('Laboratory Assistant', 'sample tracking for 15 practice samples, not clinical diagnosis', 'equipment logs for 3 training sessions'),
    ]
    rewrites = []
    jobs = []
    definitions = {
        'Vague': 'Contains unclear, non-measurable role expectations that need contextual clarification.',
        'Biased': 'Contains an explicit potentially exclusionary age, gender, or language restriction requiring contextual review.',
        'Experience_Inflated': 'Contains an experience expectation that conflicts with the stated entry level or stated technology timeline.',
    }
    for i, (role, first, second) in enumerate(roles):
        for j, fact in enumerate((first, second)):
            source = '- Helped with '+fact if j == 0 else '- Worked on '+fact
            target = '- Supported '+fact if j == 0 else '- Contributed to '+fact
            rewrites.append({'source_id':f'fictional-resume-{i+1:02}', 'original':source, 'improved':target,
                             'jd_context':f'Fictional {role} position. Keep the candidate\'s contribution accurate.'})
    neutral = ['Document existing work clearly.', 'Coordinate with the assigned supervisor.', 'Use approved processes and record decisions.', 'Review source information before sharing reports.', 'Follow the team\'s written procedures.']
    for i in range(60):
        role = roles[i % len(roles)][0]
        vague = int(i % 4 in (1, 3)); biased = int(i % 5 in (2, 4)); experience = int(i % 6 in (0, 4))
        sentences = [f'Fictional vacancy {i+1:02}: {role}.', neutral[i % len(neutral)]]
        if vague:
            sentences.append(['Be a rockstar who can do everything.', 'Be the perfect culture fit and always go above and beyond.', 'Handle unspecified tasks with unlimited flexibility.'][i % 3])
        if biased:
            sentences.append(['Young candidates only.', 'Male candidates only.', 'Native English speakers only.'][i % 3])
        if experience:
            sentences.append('This is an entry-level training role requiring no prior professional experience. Applicants must nevertheless have 12 years of professional experience.')
        else:
            sentences.append('Experience is preferred; equivalent demonstrated practice is welcome.')
        jobs.append({'source_id':f'fictional-jd-{i+1:02}', 'text':'\n'.join(sentences),
                     'Vague':str(vague),'Biased':str(biased),'Experience_Inflated':str(experience)})
    source = {'source_url':'urn:careerlens:project-authored-fictional-review-examples:v1',
              'license':'Project-authored synthetic examples, provided for use and modification in this project.',
              'synthetic':True, 'reviewed':False,
              'limitations':['AI-authored fictional practice data; no real candidates, jobs, or hiring outcomes.',
                             'Human approval of these examples does not establish real-world model quality.',
                             'Similar templates may leak phrasing across source splits; these are pipeline exercises only.']}
    bundle = {'version':1, 'source':source, 'label_definitions':definitions, 'rewrites':rewrites,'jobs':jobs}
    (root/'data'/'review_examples.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
    for name, rows, task in [('rewrites',rewrites,'resume_rewriting'),('jobs',jobs,'job_bias')]:
        raw = directory/(name+'-input.csv')
        with raw.open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        metadata = directory/(name+'-input.source.json')
        metadata.write_text(json.dumps({**source,'labels':list(definitions),'label_definitions':definitions},indent=2),encoding='utf-8')
        seed(raw,metadata,directory/(name+'-queue.csv'),task)
    print(f'Prepared {len(rewrites)} rewrite drafts and {len(jobs)} JD label drafts. All are pending human review.')


if __name__ == '__main__':
    main()
