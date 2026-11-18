"""Evidence-linked local foundation-model review, distinct from trained task models."""
from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, Field

from integrations.local_llm import local_model
from ml.skills import skill_evidence


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Finding(StrictModel):
    statement: str = Field(min_length=1, max_length=600)
    evidence: list[str] = Field(max_length=3)


class Reviewer(StrictModel):
    persona: Literal['ATS & structure', 'Technical evidence', 'Recruiter clarity']
    rubric: str = Field(max_length=400)
    strengths: list[Finding] = Field(max_length=3)
    concerns: list[Finding] = Field(max_length=3)
    recommendations: list[str] = Field(max_length=3)


class Alignment(StrictModel):
    job_quote: str = Field(min_length=1, max_length=600)
    resume_quote: str = Field(max_length=600)
    status: Literal['evidenced', 'unverified', 'not_evidenced']
    explanation: str = Field(max_length=600)


class JobFlag(StrictModel):
    phrase: str = Field(min_length=1, max_length=400)
    category: Literal['vague requirements', 'experience expectations', 'exclusionary wording']
    explanation: str = Field(max_length=600)
    suggestion: str = Field(max_length=600)


class CareerOption(StrictModel):
    role: str = Field(min_length=1, max_length=100)
    evidence: list[str] = Field(min_length=1, max_length=3)
    existing_skills: list[str] = Field(max_length=8)
    development: list[str] = Field(max_length=4)
    explanation: str = Field(max_length=500)


class Entity(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    kind: Literal['skill', 'role', 'company', 'education']
    source: Literal['resume', 'job']
    quote: str = Field(min_length=1, max_length=600)


class LocalReport(StrictModel):
    reviewers: list[Reviewer] = Field(min_length=3, max_length=3)
    alignment: list[Alignment] = Field(max_length=8)
    job_flags: list[JobFlag] = Field(max_length=5)
    career_options: list[CareerOption] = Field(max_length=3)
    entities: list[Entity] = Field(max_length=12)


REVIEW_PROMPT = '''Review the resume against the job using three distinct perspectives: ATS & structure,
Technical evidence, Recruiter clarity. Give exactly one reviewer for each perspective, with at most
one strength and one concern each, and concise recommendations. Every finding needs an EXACT
verbatim quote from either document. Omit unsupported findings; do not invent quotations.
Align up to four important job qualifications with exact resume evidence. Treat negated skills,
beginner proficiency, ownership, dates, and required/preferred context carefully. An absent quote
must have status not_evidenced and an empty resume_quote. Use unverified when proficiency or
experience cannot be established. Do not convert similarity to a score or hiring probability.
Flag only contextual vague requirements, inflated expectations, or exclusionary wording with
EXACT job phrases and concrete suggested wording. Do not estimate salary or certify fairness.
Suggest at most two exploratory career roles grounded in actual resume quotations. Existing skills
must come from the supplied existing_skills names. Development items are suggestions to learn,
never candidate qualifications. Do not infer personality, protected traits, hiring outcomes,
confidence, timelines, or guaranteed career transitions. Return up to eight named entities with
exact source quotations; each entity name must occur verbatim inside its quote. Be concise.'''


def recover_quote(quote, text):
    """Recover exact input bytes for a quotation that only changed whitespace."""
    if not quote.strip():
        return None
    if quote in text:
        return quote
    pattern = r'\s+'.join(re.escape(word) for word in quote.split())
    match = re.search(pattern, text)
    return match.group() if match else None


def review_locally(resume, job):
    evidence = skill_evidence(resume)
    raw = local_model.generate(REVIEW_PROMPT, {'resume': resume, 'job': job, 'existing_skills': list(evidence)}, LocalReport.model_json_schema())
    report = LocalReport.model_validate(raw).model_dump()
    if {item['persona'] for item in report['reviewers']} != {'ATS & structure', 'Technical evidence', 'Recruiter clarity'}:
        raise ValueError('Local reviewers must provide three distinct perspectives.')
    rejected = 0
    for reviewer in report['reviewers']:
        reviewer.update(method=f"Local Ollama / {local_model.model}", outcome='review_needed')
        for key in ('strengths', 'concerns'):
            accepted = []
            for finding in reviewer[key]:
                quotes = [recover_quote(q, resume) or recover_quote(q, job) for q in finding['evidence']]
                if quotes and all(quotes):
                    finding['evidence'] = quotes
                    accepted.append(finding)
            rejected += len(reviewer[key])-len(accepted)
            reviewer[key] = accepted
    alignment = []
    for item in report['alignment']:
        item['job_quote'] = recover_quote(item['job_quote'], job)
        quote = recover_quote(item['resume_quote'], resume) if item['status'] != 'not_evidenced' else ''
        if not item['job_quote'] or (item['status'] != 'not_evidenced' and not quote):
            rejected += 1
            continue
        item['resume_quote'] = quote
        alignment.append(item)
    report['alignment'] = alignment
    flags = []
    for flag in report['job_flags']:
        quote = recover_quote(flag['phrase'], job)
        if quote:
            flag['phrase'] = quote
            flags.append(flag)
    rejected += len(report['job_flags'])-len(flags)
    report['job_flags'] = flags
    careers = []
    for option in report['career_options']:
        quotes = [recover_quote(q, resume) for q in option['evidence']]
        if all(quotes) and all(s in evidence for s in option['existing_skills']):
            option['evidence'] = quotes
            careers.append(option)
    rejected += len(report['career_options'])-len(careers)
    report['career_options'] = careers
    entities = []
    for entity in report['entities']:
        quote = recover_quote(entity['quote'], resume if entity['source'] == 'resume' else job)
        if quote and entity['name'] in quote:
            entity['quote'] = quote
            entities.append(entity)
    rejected += len(report['entities'])-len(entities)
    report['entities'] = entities
    report.update(available=True, model=local_model.model, model_digest=local_model.status().get('digest'),
                  rejected_unverified_items=rejected,
                  note='Local foundation-model suggestions with checked source quotations. Meaning and completeness need human review; task-specific accuracy is not benchmarked.',
                  career_note='Exploratory role ideas based on supplied evidence. These are not validated GraphSAGE predictions.',
                  salary_analysis={'available': False, 'reason': 'Use the separate India salary comparison with an explicit sourced reference and compensation basis; the local wording reviewer does not estimate salary.'})
    return report
