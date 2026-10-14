"""Evidence-based requirement checks. No hiring outcome or proficiency inference."""
import re
from ml.skills import SKILL_ALIASES, _pattern, is_negated


def skill_mentions(text):
    mentions = []
    seen = set()
    for skill, aliases in SKILL_ALIASES.items():
        for alias in [skill, *aliases]:
            for match in _pattern(alias).finditer(text):
                if alias in ('Go', 'R') and match.group() != alias:
                    continue
                line_start = text.rfind('\n', 0, match.start()) + 1
                line_end = text.find('\n', match.end())
                line_end = len(text) if line_end < 0 else line_end
                quote = text[line_start:line_end].strip()
                negated = is_negated(text, match.start(), match.end())
                beginner = bool(re.search(r'\b(?:beginner|basic|learning|familiar|exposure)\b', quote, re.I))
                key = (skill, line_start, negated)
                if key not in seen:
                    seen.add(key)
                    mentions.append({'skill': skill, 'quote': quote, 'start': match.start(), 'end': match.end(),
                                     'negated': negated, 'proficiency': 'limited/self-described' if beginner else 'unspecified'})
    return sorted(mentions, key=lambda item: (item['start'], item['skill']))


def requirements_check(resume, jd):
    candidate = skill_mentions(resume)
    priorities = {}
    current = 'unspecified'
    offset = 0
    for line in jd.splitlines(keepends=True):
        if re.fullmatch(r'\s*(?:preferred(?: qualifications| skills)?|nice.to.have|desirable|bonus)\s*:?\s*', line, re.I):
            current = 'preferred'
        elif re.fullmatch(r'\s*(?:required(?: skills| qualifications)?|requirements|must.have|essential|minimum qualifications)\s*:?\s*', line, re.I):
            current = 'required'
        elif re.match(r'\s*(responsibilities|benefits|about us)\s*:?\s*$', line, re.I):
            current = 'unspecified'
        priorities[offset] = current
        offset += len(line)
    def priority_at(position):
        start = jd.rfind('\n', 0, position) + 1
        end = jd.find('\n', position)
        line = jd[start:end if end >= 0 else len(jd)]
        markers = list(re.finditer(r'\b(required|must.have|essential|preferred|nice.to.have|desirable|bonus)\b', line[:position-start], re.I))
        if not markers:
            markers = list(re.finditer(r'\b(required|must.have|essential|preferred|nice.to.have|desirable|bonus)\b', line, re.I))[:1]
        if markers:
            return 'preferred' if re.search(r'preferred|nice|desirable|bonus', markers[-1].group(), re.I) else 'required'
        return priorities.get(start, 'unspecified')
    result = []
    grouped = {}
    for mention in skill_mentions(jd):
        if mention['negated']:
            continue
        priority = priority_at(mention['start'])
        existing = grouped.get(mention['skill'])
        if not existing or (priority == 'required' and existing['priority'] != 'required'):
            grouped[mention['skill']] = {'kind': 'skill', 'requirement': mention['skill'], 'priority': priority,
                                        'job_quote': mention['quote']}
    for skill, requirement in sorted(grouped.items()):
        evidence = [m for m in candidate if m['skill'] == skill and not m['negated']]
        required_depth = bool(re.search(r'expert|advanced|proficient|production', requirement['job_quote'], re.I))
        status = 'unverified' if evidence and required_depth else 'mentioned' if evidence else 'not_evidenced'
        result.append({**requirement, 'status': status, 'evidence': [m['quote'] for m in evidence],
                       'note': 'Mentioned skills do not prove proficiency.' if evidence else 'Not evidenced in the supplied resume.'})
    pattern = r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:[\w /-]{0,35})?experience'
    declared = [(float(m.group(1)), m.group()) for m in re.finditer(pattern, resume, re.I)]
    for match in re.finditer(pattern, jd, re.I):
        requested = float(match.group(1))
        # Domain-specific years cannot safely be inferred from an overall total.
        result.append({'kind': 'experience', 'requirement': match.group(), 'priority': priority_at(match.start()),
                       'job_quote': match.group(), 'status': 'unverified',
                       'evidence': [quote for years, quote in declared],
                       'note': f'Requires {requested:g} years; verify domain, dates, and overlapping roles manually.'})
    degrees = {'Bachelor': r"\bbachelor(?:'s)?\b|\bb\.?s\.?c?\b|\bb\.?tech\b",
               'Master': r"\bmaster(?:'s)?\b|\bm\.?s\.?c?\b|\bm\.?tech\b",
               'Doctorate': r'\bph\.?d\b|\bdoctorate\b'}
    for degree, expression in degrees.items():
        if re.search(expression, jd, re.I):
            quotes = [line.strip() for line in resume.splitlines() if re.search(expression, line, re.I)]
            result.append({'kind': 'education', 'requirement': degree, 'priority': priority_at(re.search(expression, jd, re.I).start()),
                           'job_quote': next(line.strip() for line in jd.splitlines() if re.search(expression, line, re.I)),
                           'status': 'mentioned' if quotes else 'not_evidenced', 'evidence': quotes,
                           'note': 'Degree field, equivalence, and institution are not verified.'})
    return {'items': result, 'method': 'Evidence rules v1',
            'note': 'Not evidenced does not mean the candidate lacks the qualification; unverified checks need human review.'}
