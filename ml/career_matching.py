"""Personalized skill evidence and taxonomy retrieval, independent of weak GraphSAGE ranking."""
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
from ml.skills import SKILL_ALIASES, skill_evidence, is_negated, _pattern

_lock=Lock()
_cache=None
_encoder=None


def load_profiles():
    global _cache
    if _cache is None:
        import numpy as np
        root=Path('Datasets/esco')
        metadata=json.loads((root/'career_index.provenance.json').read_text(encoding='utf-8'))
        for name,key in [('profiles.json','profiles_sha256'),('career_index.npz','index_sha256')]:
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=metadata[key]:raise ValueError('Career profile/index checksum mismatch. Rebuild the public index.')
        profiles=json.loads((root/'profiles.json').read_text(encoding='utf-8'))
        with np.load(root/'career_index.npz',allow_pickle=False) as data:
            embeddings=data['embeddings'].copy();codes=data['codes'].tolist()
        if codes!=[row['code'] for row in profiles]:raise ValueError('Career index/profile ordering differs.')
        _cache=profiles,embeddings,metadata
    return _cache


def status():
    import importlib.util
    files=['profiles.json','career_index.npz','career_index.provenance.json']
    return {'available':all((Path('Datasets/esco')/name).is_file() for name in files) and all(importlib.util.find_spec(name) for name in ('torch','sentence_transformers','numpy')),
            'method':'Skill evidence + local occupation retrieval','predicts_outcomes':False}


def evidence_for(requirement,resume,evidence):
    names=[requirement]
    label=re.sub(r'\s*\([^)]*\)','',requirement).strip().casefold()
    for canonical,aliases in SKILL_ALIASES.items():
        if label in {alias.casefold() for alias in [canonical,*aliases]}:
            if canonical in evidence:return evidence[canonical]
    # Full taxonomy skill expressions can be evidenced directly; no invented aliases.
    quotes=[]
    for name in names:
        for match in _pattern(name).finditer(resume):
            if not is_negated(resume,match.start(),match.end()):
                start=resume.rfind('\n',0,match.start())+1;end=resume.find('\n',match.end())
                quote=resume[start:end if end>=0 else len(resume)].strip()
                if quote not in quotes:quotes.append(quote)
    return quotes


def recommend(resume,current_role='',target_roles=None):
    import numpy as np
    global _encoder
    with _lock:
        profiles,embeddings,metadata=load_profiles()
        by_code={row['code']:i for i,row in enumerate(profiles)}
        if current_role and current_role not in by_code:raise ValueError('Select the current occupation from the catalog.')
        if target_roles and any(code not in by_code for code in target_roles):raise ValueError('Unknown target occupation.')
        evidence=skill_evidence(resume)
        # Include non-technical evidence lines; reject entire negated/proficiency-only
        # sentences for retrieval instead of turning them into positive skill claims.
        lines=[line.strip() for line in resume.splitlines() if line.strip() and '@' not in line and not re.search(r'\b(?:no|not|never|without|lack|lacking)\b',line,re.I)]
        if not evidence and not any(len(line.split())>=4 and (re.match(r'^[-•*]',line) or re.match(r'(?i)^skills\s*:',line)) for line in lines):
            return {'available':False,'reason':'Add demonstrated skills or work evidence before requesting role matches.'}
        query='Skills: '+', '.join(evidence)+'. '+('Current occupation: '+profiles[by_code[current_role]]['title']+'. ' if current_role else '')+' '.join(lines[:20])
        if _encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(4)
            _encoder=SentenceTransformer('Datasets/models/minilm',device='cpu',local_files_only=True,trust_remote_code=False)
        ids=_encoder.tokenizer.encode(query,add_special_tokens=False)
        if len(ids)>_encoder.max_seq_length-2:
            # Retain the skill summary and explicitly selected role; bounded work
            # excerpts remain retrieval context, never a complete resume assessment.
            query=_encoder.tokenizer.decode(ids[:_encoder.max_seq_length-2],skip_special_tokens=True)
            bounded=True
        else:bounded=False
        vector=_encoder.encode([query],normalize_embeddings=True,show_progress_bar=False)[0]
        scores=embeddings@vector
        candidates=[by_code[code] for code in target_roles] if target_roles else [int(i) for i in np.argsort(-scores)[:20] if profiles[int(i)]['code']!=current_role]
        results=[]
        for i in candidates:
            profile=profiles[i];skills=[]
            for group in ('essential','optional'):
                for skill in profile[group]:
                    quotes=evidence_for(skill['name'],resume,evidence)
                    skills.append({**skill,'priority':group,'evidence':quotes,'status':'mentioned' if quotes else 'not_evidenced'})
            matched=[s for s in skills if s['evidence']]
            results.append({'code':profile['code'],'role':profile['title'],'source_uri':profile['uri'],
                'relatedness':round(float(scores[i]),4),'matched_skills':matched,
                'development':[s for s in skills if not s['evidence'] and s['priority']=='essential'],
                'optional_skills':[s for s in skills if not s['evidence'] and s['priority']=='optional'],
                'essential_count':len(profile['essential']),'matched_essential_count':sum(s['priority']=='essential' for s in matched)})
        # Keep retrieval order. Counting keywords across differently sized taxonomy
        # profiles can promote unrelated occupations above the user's actual field.
        return {'available':True,'roles':results[:5],'existing_skills':list(evidence),'current_role':current_role or None,
                'source_version':'ESCO v1.1.2','retrieval_excerpt_bounded':bounded,
                'note':'Matches use your actual resume evidence and sourced essential/optional occupation skills. Not evidenced does not mean absent; mentions do not prove proficiency. Education, licences, experience depth and Indian employer requirements need confirmation. Relatedness is not hiring probability or a career-outcome forecast. The European taxonomy is used as a skills reference, not Indian transition statistics.'}
