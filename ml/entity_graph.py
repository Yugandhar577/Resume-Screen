"""Versioned local NER graph: exact spans and observed relations, no inferred ownership."""
import re
from ml.skills import SKILL_ALIASES, _pattern

NER_PACKAGE = 'en_core_web_sm'
NER_VERSION = '3.8.0'


class EntityGraph:
    def __init__(self):
        import spacy
        self.nlp = spacy.load(NER_PACKAGE, disable=['parser', 'lemmatizer'])
        if self.nlp.meta['version'] != NER_VERSION:
            raise ValueError('Install the pinned en_core_web_sm 3.8.0 model to reproduce entity graphs.')

    def extract(self, text):
        # Keep input offsets exact; never report dictionary aliases as verbatim source names.
        entities = []
        for canonical, aliases in SKILL_ALIASES.items():
            for alias in [canonical, *aliases]:
                for match in _pattern(alias).finditer(text):
                    if alias in ('Go', 'R') and match.group() != alias:
                        continue
                    entities.append({'name': canonical, 'kind': 'SKILL', 'start': match.start(), 'end': match.end(), 'method': 'canonical vocabulary'})
        for match in re.finditer(r'\b\d+\+?\s+years?(?:\s+of)?\s+experience\b|\b(?:bachelor|master|doctorate|degree|diploma)\b', text, re.I):
            entities.append({'name': match.group(), 'kind': 'QUALIFICATION', 'start': match.start(), 'end': match.end(), 'method': 'pattern'})
        from ml.semantic_matching import PUBLIC_ROLE_FAMILIES
        for role in PUBLIC_ROLE_FAMILIES:
            for match in re.finditer(r'(?<!\w)'+re.escape(role)+r'(?!\w)', text, re.I):
                entities.append({'name': match.group(), 'kind': 'ROLE', 'start': match.start(), 'end': match.end(), 'method': 'role vocabulary'})
        for entity in self.nlp(text).ents:
            if entity.label_ in ('ORG', 'PRODUCT', 'DATE'):
                entities.append({'name': entity.text, 'kind': entity.label_, 'start': entity.start_char, 'end': entity.end_char, 'method': 'pretrained spaCy NER'})
        unique = {}
        for entity in sorted(entities, key=lambda row:(row['start'], row['end'], row['kind'])):
            key = (entity['start'], entity['end'], entity['kind'])
            unique[key] = {**entity, 'quote': text[entity['start']:entity['end']]}
        return list(unique.values())

    def nodes(self, text):
        lines = []
        offset = 0
        for line in text.splitlines(keepends=True):
            if line.strip() and '@' not in line:
                for start in range(0, len(line), 400):
                    quote = line[start:start+400].strip()
                    if quote:
                        exact_start = offset+start+len(line[start:start+400])-len(line[start:start+400].lstrip())
                        lines.append({'name': quote, 'kind': 'TEXT', 'quote': quote, 'start': exact_start, 'end': exact_start+len(quote), 'method': 'observed source chunk'})
            offset += len(line)
        nodes = lines[:16]
        if not nodes:
            raise ValueError('No usable source lines for entity graph.')
        # Entity nodes only from encoded source chunks, bounded to 32 total nodes.
        for entity in self.extract(text):
            if any(entity['start'] >= line['start'] and entity['end'] <= line['end'] for line in nodes if line['kind']=='TEXT'):
                nodes.append(entity)
            if len(nodes) == 32:
                break
        relations = []
        for i, first in enumerate(nodes):
            for j in range(i+1, len(nodes)):
                second = nodes[j]
                if first['kind']=='TEXT' and second['kind']=='TEXT' and j==i+1:
                    relations.append({'source': i, 'target': j, 'kind': 'adjacent_source_chunks'})
                elif first['kind']=='TEXT' and second['start']>=first['start'] and second['end']<=first['end']:
                    relations.append({'source': i, 'target': j, 'kind': 'source_contains_entity'})
        return nodes, relations
