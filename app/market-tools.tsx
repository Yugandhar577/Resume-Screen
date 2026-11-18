'use client'
import {useState,useRef} from 'react'

type Reference={id:string;role:string;experience_min:number;experience_max:number;annual_low:number;annual_high:number;currency:string;pay_basis:string;title:string;url:string;as_of:string;geography:string;method:string;granularity:string;user_provided?:boolean}
type SalaryReport={comparison:string;reason?:string;note:string;location_note:string;date_note?:string;boundary_note?:string;experience_note?:string;reference:Reference;offers:{quote:string;valid:boolean;currency?:string;annual_low?:number;annual_high?:number;reason?:string}[]}
type Skill={name:string;uri:string;priority:string;evidence:string[];status:string}
type CareerReport={available:boolean;reason?:string;note?:string;roles?:{code:string;role:string;source_uri:string;relatedness:number;matched_skills:Skill[];development:Skill[];optional_skills:Skill[];essential_count:number;matched_essential_count:number}[]}
type Role={code:string;title:string}

export default function MarketTools({resume,jobDescription,headers}:{resume:string;jobDescription:string;headers:Record<string,string>}) {
  const [references,setReferences]=useState<Reference[]>([])
  const [referenceId,setReferenceId]=useState('')
  const [custom,setCustom]=useState<Record<string,unknown>|null>(null)
  const [offer,setOffer]=useState('')
  const [years,setYears]=useState('')
  const [city,setCity]=useState('India')
  const [basis,setBasis]=useState('unspecified')
  const [salary,setSalary]=useState<SalaryReport|null>(null)
  const [career,setCareer]=useState<CareerReport|null>(null)
  const [roleQuery,setRoleQuery]=useState('')
  const [roles,setRoles]=useState<Role[]>([])
  const [currentRole,setCurrentRole]=useState('')
  const [targetQuery,setTargetQuery]=useState('')
  const [targetRoles,setTargetRoles]=useState<Role[]>([])
  const [targetRole,setTargetRole]=useState('')
  const [busy,setBusy]=useState('')
  const [error,setError]=useState('')
  const [salaryError,setSalaryError]=useState('')
  const fileInput=useRef<HTMLInputElement>(null)
  const request=async(url:string,body?:unknown)=>{
    const response=await fetch(url,{method:body?'POST':'GET',headers:{...headers,...(body?{'Content-Type':'application/json'}:{})},...(body?{body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(90000)})
    const data=await response.json()
    if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:Array.isArray(data.detail)?data.detail.map((d:{msg:string})=>d.msg).join('; '):'Could not complete the request.')
    return data
  }
  const loadReferences=async()=>{
    setBusy('references');setSalaryError('')
    try{const data=await request('/api/salary-references');setReferences(data.references);setReferenceId(data.references.find((r:Reference)=>jobDescription.toLowerCase().includes(r.role.toLowerCase()))?.id||data.references[0]?.id||'')}
    catch(e){setSalaryError(e instanceof Error?e.message:'References unavailable')}
    finally{setBusy('')}
  }
  const compare=async()=>{
    setBusy('salary');setSalaryError('');setSalary(null)
    try{setSalary(await request('/api/salary-compare',{offerText:offer.trim()||jobDescription,referenceId,experienceYears:Number(years),city,payBasis:basis,...(custom?{reference:custom}:{})}))}
    catch(e){setSalaryError(e instanceof Error?e.message:'Comparison failed')}
    finally{setBusy('')}
  }
  const match=async()=>{
    setBusy('career');setError('');setCareer(null)
    try{setCareer(await request('/api/career-recommendations',{resume,currentRoleCode:currentRole,targetRoleCodes:targetRole?[targetRole]:[]}))}
    catch(e){setError(e instanceof Error?e.message:'Career matching failed')}
    finally{setBusy('')}
  }
  const findRoles=async()=>{
    setBusy('roles');setError('');setCurrentRole('')
    try{const data=await request('/api/career-roles?query='+encodeURIComponent(roleQuery));setRoles(data.roles);if(!data.roles.length)setError('No matching occupation in the catalog.')}
    catch(e){setError(e instanceof Error?e.message:'Search failed')}
    finally{setBusy('')}
  }
  const findTargets=async()=>{
    setBusy('targets');setError('');setTargetRole('');setCareer(null)
    try{const data=await request('/api/career-roles?query='+encodeURIComponent(targetQuery));setTargetRoles(data.roles);if(!data.roles.length)setError('No matching target occupation in the catalog.')}
    catch(e){setError(e instanceof Error?e.message:'Search failed')}
    finally{setBusy('')}
  }
  const importReference=async(file:File)=>{
    try{if(file.size>30000)throw new Error('Benchmark files must be under 30 KB.');const data=JSON.parse(await file.text());if(typeof data.role!=='string'||typeof data.title!=='string')throw new Error('Expected one benchmark object with role, source, date and annual INR ranges.');setCustom(data);setSalary(null);setSalaryError('')}
    catch(e){setSalaryError(e instanceof Error?e.message:'Invalid benchmark file')}
  }
  const money=(value:number|undefined,currency='INR')=>value===undefined?'Unavailable':new Intl.NumberFormat('en-IN',{style:'currency',currency,maximumFractionDigits:0}).format(value)
  const field='block w-full rounded border border-neutral-200 p-3 mt-2 bg-white'
  return <div className="space-y-5">
    <section className="border border-neutral-200 rounded-lg p-5 space-y-4">
      <h3 className="font-medium">Career Matches From Your Skills</h3>
      <p className="text-sm text-neutral-500">Compare your demonstrated skills with sourced occupation requirements. Current role is optional. This workflow does not use the experimental historical transition ranking.</p>
      <details className="text-sm"><summary className="cursor-pointer">Confirm current occupation (optional)</summary><div className="mt-3 space-y-3"><label>Current role search<input aria-label="Current role search" className={field} value={roleQuery} maxLength={100} onChange={e=>setRoleQuery(e.target.value)}/></label><button disabled={!!busy||roleQuery.trim().length<2} onClick={findRoles} className="border border-neutral-200 px-4 py-2 rounded disabled:opacity-40">Search current roles</button>{roles.length>0&&<label className="block">Current role<select aria-label="Confirmed current role" className={field} value={currentRole} onChange={e=>{setCurrentRole(e.target.value);setCareer(null)}}><option value="">No current role selected</option>{roles.map(r=><option key={r.code} value={r.code}>{r.title}</option>)}</select></label>}</div></details>
      <details className="text-sm"><summary className="cursor-pointer">Compare a specific target role (optional)</summary><div className="mt-3 space-y-3"><label>Target role search<input aria-label="Target role search" className={field} value={targetQuery} maxLength={100} onChange={e=>setTargetQuery(e.target.value)}/></label><button disabled={!!busy||targetQuery.trim().length<2} onClick={findTargets} className="border border-neutral-200 px-4 py-2 rounded disabled:opacity-40">Search target roles</button>{targetRoles.length>0&&<label className="block">Target role<select aria-label="Confirmed target role" className={field} value={targetRole} onChange={e=>{setTargetRole(e.target.value);setCareer(null)}}><option value="">Find related occupations automatically</option>{targetRoles.map(r=><option key={r.code} value={r.code}>{r.title}</option>)}</select></label>}</div></details>
      <button onClick={match} disabled={!!busy} className="bg-black text-white rounded px-4 py-2 text-sm disabled:opacity-40">{busy==='career'?'Matching your evidence…':'Find career matches'}</button>
      {error&&<p role="alert" className="text-sm">{error}</p>}
      {career&&<><p className="text-xs text-neutral-500">{career.available?career.note:career.reason}</p>{career.roles?.map(role=><details key={role.code} className="border-t border-neutral-100 pt-3 text-sm"><summary className="font-medium cursor-pointer">{role.role} · {role.matched_essential_count}/{role.essential_count} essential skills evidenced</summary><p className="mt-3 text-xs text-neutral-500">Source: European Commission ESCO v1.1.2 · Text relatedness {role.relatedness.toFixed(3)} · ESCO {role.code}. This is not readiness or hiring probability.</p><h4 className="mt-4 font-medium">Evidence in your resume</h4>{role.matched_skills.length?role.matched_skills.map(skill=><div key={skill.uri} className="mt-3"><p>{skill.name} · {skill.priority}</p>{skill.evidence.map((quote,i)=><blockquote key={i} className="border-l-2 pl-3 text-neutral-600 mt-1">{quote}</blockquote>)}</div>):<p className="mt-2 text-neutral-500">No direct taxonomy skill matches; inspect the role before pursuing it.</p>}<a href={`https://esco.ec.europa.eu/en/classification/occupation?uri=${encodeURIComponent(role.source_uri)}`} target="_blank" rel="noopener noreferrer" className="block underline mt-3">Occupation requirements source</a><h4 className="mt-4 font-medium">Requirements to verify or develop</h4><ul className="list-disc pl-5 mt-2 space-y-1">{role.development.map(skill=><li key={skill.uri}>{skill.name}</li>)}</ul>{role.optional_skills.length>0&&<details className="mt-4"><summary className="cursor-pointer">Optional skills</summary><ul className="list-disc pl-5 mt-2">{role.optional_skills.map(skill=><li key={skill.uri}>{skill.name}</li>)}</ul></details>}</details>)}</>}
    </section>
    <section className="border border-neutral-200 rounded-lg p-5 space-y-4">
      <h3 className="font-medium">India Salary Comparison</h3>
      <p className="text-sm text-neutral-500">Use published India references or a benchmark you have permission to use. Figures retain their source year, coverage and compensation basis; no foreign-market or city uplift is invented.</p>
      {!references.length&&<button onClick={loadReferences} disabled={!!busy} className="border border-neutral-200 px-4 py-2 rounded text-sm disabled:opacity-40">Load India salary references</button>}
      {references.length>0&&<label className="block text-sm">Salary reference<select aria-label="Salary reference" className={field} value={custom?'custom':referenceId} onChange={e=>{setCustom(null);setReferenceId(e.target.value);setSalary(null)}}>{custom&&<option value="custom">User benchmark: {String(custom.role)}</option>}{references.map(r=><option key={r.id} value={r.id}>{r.role} · {r.experience_max===60?`${r.experience_min}+`:`${r.experience_min}–${r.experience_max}`} years · {r.granularity} · {r.as_of}</option>)}</select></label>}
      <div className="grid sm:grid-cols-2 gap-4"><label className="text-sm">Confirmed years of experience<input aria-label="Salary experience years" type="number" min="0" max="60" step="0.5" value={years} onChange={e=>{setYears(e.target.value);setSalary(null)}} className={field}/></label><label className="text-sm">Indian location<input aria-label="Salary location" value={city} maxLength={100} onChange={e=>{setCity(e.target.value);setSalary(null)}} className={field}/></label></div>
      <label className="block text-sm">Offer salary text<input aria-label="Offer salary text" value={offer} maxLength={1500} onChange={e=>{setOffer(e.target.value);setSalary(null)}} className={field} placeholder="For example: INR 8–12 LPA CTC"/><span className="block text-xs text-neutral-500 mt-2">Leave blank to extract salary from the analyzed job description. Enter one range; separate bonus and equity components.</span></label>
      <label className="block text-sm">Compensation basis<select aria-label="Compensation basis" value={basis} onChange={e=>{setBasis(e.target.value);setSalary(null)}} className={field}><option value="unspecified">Use explicitly stated basis, otherwise unknown</option><option value="ctc">I confirm this is annual CTC</option><option value="base">I confirm this is base salary</option></select></label>
      <div className="flex flex-wrap gap-3"><button disabled={!!busy||(!referenceId&&!custom)||years===''||Number(years)<0||Number(years)>60} onClick={compare} className="bg-black text-white px-4 py-2 rounded text-sm disabled:opacity-40">{busy==='salary'?'Comparing…':'Compare salary'}</button><button onClick={()=>fileInput.current?.click()} className="text-sm underline">Use a local benchmark JSON</button><input ref={fileInput} aria-label="Import salary benchmark" type="file" accept=".json,application/json" className="hidden" onChange={e=>{if(e.target.files?.[0])importReference(e.target.files[0]);e.target.value=''}}/></div>
      {custom&&<p className="text-xs text-neutral-500">Using {String(custom.title)}. User-supplied source values are not independently verified. Import stays in this tab and is not saved.</p>}
      {salaryError&&<p role="alert" className="text-sm">{salaryError}</p>}
      {salary&&<div className="text-sm space-y-3 border-t border-neutral-100 pt-4"><p role="status" className="font-medium">{salary.comparison.replaceAll('_',' ')}</p>{salary.reason&&<p>{salary.reason}</p>}<p>Reference: {money(salary.reference.annual_low)}–{money(salary.reference.annual_high)} annually · {salary.reference.pay_basis.toUpperCase()}</p>{salary.offers.map((o,i)=><p key={i}>{o.valid?`Offer normalized: ${money(o.annual_low,o.currency)}–${money(o.annual_high,o.currency)} annually`:o.reason}<span className="block text-xs text-neutral-500">Extracted: “{o.quote}”</span></p>)}<a href={salary.reference.url} target="_blank" rel="noopener noreferrer" className="underline block">{salary.reference.title} · {salary.reference.as_of}</a><p className="text-xs text-neutral-500">{salary.reference.geography}. {salary.reference.method}. {salary.location_note} {salary.date_note} {salary.boundary_note} {salary.experience_note} {salary.note}</p></div>}
    </section>
  </div>
}
