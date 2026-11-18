"""Sourced India salary references and explicit unit/basis comparisons. No AI estimates."""
from datetime import date
import json
import math
from pathlib import Path
import re

REFERENCES=Path(__file__).resolve().parents[1]/'data/india_salary_references.json'


def catalog():
    source=json.loads(REFERENCES.read_text(encoding='utf-8'))
    rows=[]
    for i,(role,minimum,maximum,low,high) in enumerate(source['role_ranges']):
        rows.append({'id':f'syncwell-{i}','role':role,'experience_min':minimum,'experience_max':maximum,
                     'annual_low':round(low*100000),'annual_high':round(high*100000),'granularity':'role',**source['sources']['syncwell-2025']})
    for i,(role,ranges) in enumerate(source['industry_ranges'].items()):
        for j,((minimum,maximum),(low,high)) in enumerate(zip(source['experience_bands'],ranges)):
            rows.append({'id':f'foundit-{i}-{j}','role':role,'experience_min':minimum,'experience_max':maximum,
                         'annual_low':round(low*100000),'annual_high':round(high*100000),'granularity':'industry',**source['sources']['foundit-oct2025']})
    return rows


def extract_offers(text):
    # Currency or a lakh-per-annum suffix is mandatory: experience years are never salaries.
    number=r'\d+(?:,\d{2,3})*(?:\.\d+)?'
    scale=r'(?:lpa|lakhs?|lacs?|crores?|k)'
    currency=r'(?:₹|INR\b|Rs\.?\s*|USD\b|\$|EUR\b|€|GBP\b|£)'
    expression=rf'(?P<currency>{currency})?\s*(?P<low>{number})\s*(?P<scale1>{scale})?\s*(?:[-–—]\s*(?:{currency})?\s*(?P<high>{number})\s*(?P<scale2>{scale})?)?\s*(?P<period>per\s+(?:month|year|annum)|/(?:mo(?:nth)?|year|yr)|monthly|annually|annual|p\.?a\.?)?'
    offers=[]
    for match in re.finditer(expression,text,re.I):
        raw=match.group().strip();values=match.groupdict();unit=(values['scale2'] or values['scale1'] or '').lower()
        if not values['currency'] and unit!='lpa':continue
        if not values['period'] and unit!='lpa':continue
        if len(offers)>=5:break
        prefix=text[max(0,match.start()-20):match.start()]
        suffix=text[match.end():match.end()+3]
        if prefix.endswith('-') or re.search(r'\b(?:up to|starting at|from)\s*$',prefix,re.I) or suffix.lstrip().startswith('+'):
            offers.append({'quote':raw,'valid':False,'reason':'Negative or open-ended salary cannot be treated as a complete bounded offer.'});continue
        c=(values['currency'] or 'INR').upper()
        code='USD' if '$' in c or 'USD' in c else 'EUR' if '€' in c or 'EUR' in c else 'GBP' if '£' in c or 'GBP' in c else 'INR'
        found_currency=re.findall(currency,raw,re.I)
        currency_codes={'USD' if '$' in item or 'USD' in item.upper() else 'EUR' if '€' in item or 'EUR' in item.upper() else 'GBP' if '£' in item or 'GBP' in item.upper() else 'INR' for item in found_currency}
        if len(currency_codes)>1:
            offers.append({'quote':raw,'valid':False,'reason':'Mixed currencies in one range.'});continue
        multiplier=lambda value:10000000 if value.startswith('crore') else 100000 if value.startswith(('lac','lakh','lpa')) else 1000 if value=='k' else 1
        shared=values['scale2'] or values['scale1'] or ''
        low=float(values['low'].replace(',',''))*multiplier((values['scale1'] or shared).lower())
        high=float((values['high'] or values['low']).replace(',',''))*multiplier((values['scale2'] or shared).lower())
        monthly=bool(re.search(r'month|/mo',values['period'] or '',re.I))
        if unit=='lpa' and monthly:
            offers.append({'quote':raw,'valid':False,'reason':'Conflicting annual and monthly units.'});continue
        if not 0<low<=high<1e11 or not all(map(math.isfinite,[low,high])):
            offers.append({'quote':raw,'valid':False,'reason':'Invalid salary range.'});continue
        context=text[max(0,match.start()-40):min(len(text),match.end()+40)]
        basis='ctc' if re.search(r'\bctc\b|cost to company',context,re.I) else 'base' if re.search(r'\bbase\b|fixed pay|basic salary',context,re.I) else 'unspecified'
        offers.append({'quote':raw,'valid':True,'currency':code,'annual_low':round(low*(12 if monthly else 1),2),
                       'annual_high':round(high*(12 if monthly else 1),2),'period':'monthly' if monthly else 'annual','pay_basis':basis})
    return offers


def compare(text,reference_id,experience_years,city='India',pay_basis='unspecified',reference=None):
    reference=reference or next((row for row in catalog() if row['id']==reference_id),None)
    if not reference:raise ValueError('Choose a sourced salary reference.')
    offers=extract_offers(text)
    result={'available':True,'reference':reference,'offers':offers,'comparison':'needs_offer',
            'location_note':f'{city or "India"}: this source has no verified city-specific adjustment.',
            'note':'Published reference ranges are not a personal salary prediction, a fair-pay verdict, or individual percentiles. Experience alone does not establish job equivalence.'}
    compared_years=math.floor(experience_years) if reference.get('granularity')=='industry' else experience_years
    if reference.get('granularity')=='industry':
        result['experience_note']='Industry brackets use completed years; fractional years are rounded down for bracket selection.'
    if not reference['experience_min']<=compared_years<=reference['experience_max']:
        return {**result,'comparison':'outside_experience_range','reason':'Your confirmed experience is outside this reference bracket; choose another reference.'}
    if experience_years==15 and reference_id.startswith('foundit'):
        result['boundary_note']='The source overlaps its 11–15 and 15+ labels. The selected reference determines the bracket; no interpolation is applied.'
    if len(offers)!=1 or not offers[0].get('valid'):
        return {**result,'reason':'Enter one complete salary range with currency and pay period. Separate base, bonus and equity components.'}
    offer=offers[0];basis=pay_basis if pay_basis!='unspecified' else offer['pay_basis']
    if pay_basis!='unspecified' and offer['pay_basis']!='unspecified' and pay_basis!=offer['pay_basis']:
        return {**result,'comparison':'not_comparable','reason':'Confirmed compensation basis conflicts with the offer text. Correct the text or selection.'}
    if offer['currency']!=reference['currency']:
        return {**result,'comparison':'not_comparable','reason':'Currencies differ. No exchange-rate conversion or foreign-market substitution was applied.'}
    if reference['pay_basis']=='unspecified' or basis=='unspecified':
        return {**result,'comparison':'reference_only','reason':'Compensation basis is not established for both values. No salary-mismatch conclusion is made.'}
    if basis!=reference['pay_basis']:
        return {**result,'comparison':'not_comparable','reason':'Base pay and CTC are different compensation measures.'}
    old=int(reference['as_of'][:4])<date.today().year
    result['date_note']='Historical source; no inflation or present-day update is inferred.' if old else 'Use the source date when interpreting this reference.'
    low,high=reference['annual_low'],reference['annual_high']
    result['comparison']='below_reference' if offer['annual_high']<low else 'above_reference' if offer['annual_low']>high else 'overlaps_reference'
    return result
