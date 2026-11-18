"""Explicit download of public ESCO skills profiles; no candidate data or API key."""
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = 'https://ec.europa.eu/esco/api/search'


def page(index):
    cache=Path('Datasets/esco/profile-pages-v112')/f'{index}.json'
    if cache.is_file():
        try:
            saved=json.loads(cache.read_text(encoding='utf-8'))
            return saved['rows'],saved['total'],saved['sha256']
        except (ValueError,KeyError):pass
    parameters={'text':'','type':'occupation','language':'en','limit':50,'offset':index,'full':'true','selectedVersion':'v1.1.2'}
    raw=bytearray()
    session=requests.Session()
    session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=[429,502,503,504],allowed_methods=['GET'])))
    with session,session.get(URL,params=parameters,timeout=(15,90),stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(65536):
            raw.extend(chunk)
            if len(raw)>24*1024**2:raise ValueError('ESCO page exceeds acquisition limit.')
    data=json.loads(raw)
    profiles=[]
    for row in data['_embedded']['results']:
        links=row.get('_links',{})
        def skills(name):return [{'name':s['title'],'uri':s['uri']} for s in links.get(name,[]) if s.get('title') and s.get('uri')]
        profiles.append({'code':row['code'],'title':row['preferredLabel']['en'],'uri':row['uri'],
            'description':row.get('description',{}).get('en',{}).get('literal',''),
            'essential':skills('hasEssentialSkill'),'optional':skills('hasOptionalSkill')})
    digest=hashlib.sha256(raw).hexdigest()
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_text(json.dumps({'rows':profiles,'total':data['total'],'sha256':digest}),encoding='utf-8')
    return profiles,data['total'],digest


def main():
    root=Path('Datasets/esco');root.mkdir(parents=True,exist_ok=True)
    first,total,digest=page(0);all_rows=first;hashes=[digest]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i,(rows,count,digest) in enumerate(pool.map(page,range(1,math.ceil(total/50))),1):
            if count!=total:raise ValueError('Taxonomy changed during acquisition; retry.')
            all_rows.extend(rows);hashes.append(digest)
            if i%10==0:print(f'Acquired {len(all_rows)}/{total} public profiles',flush=True)
    if len(all_rows)!=total or len({r['code'] for r in all_rows})!=total:raise ValueError('Incomplete or duplicate ESCO catalog.')
    content=json.dumps(sorted(all_rows,key=lambda r:r['code']),ensure_ascii=False,indent=2)
    target=root/'profiles.json';target.write_text(content,encoding='utf-8')
    metadata={'source':URL,'version':'v1.1.2','attribution':'European Commission ESCO',
              'reuse_notice':'https://commission.europa.eu/legal-notice_en','page_sha256':hashes,
              'profiles_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'profiles':total,
              'limitations':['European occupational taxonomy, not observed Indian career outcomes.','Essential/optional requirements describe occupations; candidate evidence needs confirmation.']}
    (root/'profiles.provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Completed {total} occupation-to-skills profiles.',flush=True)


if __name__=='__main__':main()
