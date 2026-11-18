"""Reviewable, constrained factual rewrites and source-preserving section ordering."""
import hashlib
import re
from collections import Counter
from ml.resume_optimizer import ResumeOptimizer

RULES = [
    ('worked on ', 'Contributed to ', 'Describe contribution without claiming ownership'),
    ('helped with ', 'Supported ', 'Use a concise action verb while preserving supporting involvement'),
    ('assisted with ', 'Supported ', 'Preserve supporting involvement'),
    ('was responsible for ', 'Handled ', 'Remove introductory wording'),
    ('responsible for ', 'Handled ', 'Start with a direct action'),
    ('worked with the team to ', 'Collaborated with the team to ', 'Make collaboration explicit'),
]
TEMPLATES = {
    'original': {'label': 'Keep current order', 'order': []},
    'engineering': {'label': 'Engineering', 'order': ['summary', 'skills', 'experience', 'projects', 'education', 'certifications']},
    'data': {'label': 'Data & analytics', 'order': ['summary', 'skills', 'projects', 'experience', 'education', 'certifications']},
    'general': {'label': 'General professional', 'order': ['summary', 'experience', 'skills', 'education', 'projects', 'certifications']},
    'healthcare': {'label': 'Healthcare', 'order': ['summary', 'certifications', 'education', 'experience', 'skills', 'projects']},
    'education': {'label': 'Education & research', 'order': ['summary', 'education', 'experience', 'projects', 'certifications', 'skills']},
    'finance': {'label': 'Finance & operations', 'order': ['summary', 'certifications', 'experience', 'skills', 'education', 'projects']},
}


def constrained_rewrite(line):
    match = re.match(r'^(\s*[-•●▪*]\s+)(.*)$', line)
    if not match:
        return None
    marker, content = match.groups()
    for source, target, reason in RULES:
        if content.casefold().startswith(source):
            return marker + target + content[len(source):], reason
    return None


def validate_rewrite(source, proposed):
    expected = constrained_rewrite(source)
    valid = bool(expected and proposed == expected[0])
    return {'valid': valid, 'method': 'Constrained semantic transformation v1',
            'issues': [] if valid else ['The edit is not an approved fact-preserving transformation.'],
            'scope': 'Only the introductory action phrase may change; all remaining facts, qualifiers, negation, and metrics must be identical.'}


def proposals(resume):
    result = []
    for index, line in enumerate(resume.splitlines()):
        rewrite = constrained_rewrite(line)
        if rewrite:
            after, reason = rewrite
            result.append({'id': hashlib.sha256(f'{index}:{line}'.encode()).hexdigest()[:20],
                           'line': index + 1, 'original': line, 'proposed': after, 'evidence': line,
                           'reason': reason, 'validation': validate_rewrite(line, after)})
    return result


def reorder_sections(resume, template):
    if template not in TEMPLATES:
        raise ValueError('Unknown role template.')
    order = TEMPLATES[template]['order']
    if not order:
        return resume
    headings = {'professional experience': 'experience', 'work experience': 'experience',
                'professional summary': 'summary', 'technical skills': 'skills'}
    blocks, prefix, block, current = [], [], [], None
    for line in resume.splitlines(keepends=True):
        name = line.strip().rstrip(':').casefold()
        name = headings.get(name, name)
        if name in {'summary', 'skills', 'experience', 'projects', 'education', 'certifications'}:
            if current is not None:
                blocks.append((current, ''.join(block)))
            current, block = name, [line]
        elif current is None:
            prefix.append(line)
        else:
            block.append(line)
    if current is not None:
        blocks.append((current, ''.join(block)))
    if len(blocks) < 2:
        return resume
    blocks.sort(key=lambda item: order.index(item[0]) if item[0] in order else len(order))
    result = ''.join(prefix) + '\n'.join(body.rstrip('\n') for _, body in blocks)
    if Counter(result.split()) != Counter(resume.split()):
        raise ValueError('Section reordering changed content.')
    return result


def apply_edits(resume, accepted_ids, template='original'):
    edits = {edit['id']: edit for edit in proposals(resume)}
    if len(set(accepted_ids)) != len(accepted_ids) or any(key not in edits for key in accepted_ids):
        raise ValueError('An edit is stale, duplicated, or does not belong to this resume.')
    lines = resume.splitlines()
    for key in accepted_ids:
        edit = edits[key]
        if not validate_rewrite(edit['original'], edit['proposed'])['valid']:
            raise ValueError('An edit failed factual validation.')
        lines[edit['line'] - 1] = edit['proposed']
    result = reorder_sections('\n'.join(lines), template)
    cleaned = ResumeOptimizer().prepare_resume(result)
    return {**cleaned, 'method': 'Reviewed_Constrained_Rewrite', 'accepted_ids': accepted_ids,
            'template': template, 'applied_edits': [edits[key] for key in accepted_ids],
            'note': 'Only accepted, constrained rewrites and selected section ordering were applied. No missing skills or facts were added.'}
