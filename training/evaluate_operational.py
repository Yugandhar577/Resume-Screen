"""Fresh authored stress cases, not human labels or real hiring validation."""
import hashlib
import json
from pathlib import Path
from ml.rewrite_workflow import RULES,constrained_rewrite,validate_rewrite
from ml.local_rewriter import check_facts
from ml.requirements import requirements_check


def main():
    cases=[]
    for i,(prefix,target,reason) in enumerate(RULES):
        for count in (3,17,89,240):
            for tech in ('Python','SQL','React','Excel'):
                original=f'- {prefix.capitalize()}{tech} reporting for {count} users at Cedar Labs in 2026, not production'
                proposed=constrained_rewrite(original)[0]
                valid=validate_rewrite(original,proposed)['valid'] and check_facts(original,proposed)['valid']
                corruptions=[proposed.replace(str(count),str(count+1)),proposed.replace('not ',''),proposed.replace('Cedar','Other'),proposed.replace(tech,'AWS'),proposed.replace('users','teams')]
                rejected=all(not check_facts(original,changed)['valid'] for changed in corruptions)
                cases.append({'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'valid_transform':valid,'corruptions_rejected':rejected})
    qualification=[]
    for tech in ('Python','SQL','React','Accounting','Teaching','Patient Care'):
        for resume,expected in [(f'Skills: {tech}','mentioned'),(f'No {tech} experience.','not_evidenced'),('Unrelated experience','not_evidenced')]:
            items=requirements_check(resume,'Required: '+tech)['items']
            qualification.append(any(row['kind']=='skill' and row['requirement']==tech and row['status']==expected for row in items))
    report={'version':'operational-stress-v1','source':'Project-authored fictional fixtures created after the initial training runs',
            'rewriting_cases':len(cases),'valid_transform_passes':sum(c['valid_transform'] for c in cases),
            'adversarial_changes':len(cases)*5,'adversarial_changes_rejected':sum(c['corruptions_rejected'] for c in cases)*5,
            'qualification_cases':len(qualification),'qualification_cases_passed':sum(qualification),
            'limitations':['Tests approved transformation invariants and selected evidence-rule cases, not human writing quality.','No actual candidate data, relevance judgments, hiring probabilities or career outcomes are validated.'],
            'case_source_hashes':[c['source_sha256'] for c in cases]}
    output=Path('evaluation');output.mkdir(exist_ok=True)
    (output/'operational_stress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='case_source_hashes'},indent=2))
    if not all(c['valid_transform'] and c['corruptions_rejected'] for c in cases) or not all(qualification):raise SystemExit('Stress evaluation failed.')


if __name__=='__main__':main()
