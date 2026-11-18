'use client'
import { useState } from 'react'

type Role = { code: string; title: string; uri: string; activation?: number }
type Report = { available: boolean; reason?: string; note?: string; roles?: Role[] }

export default function CareerExplorer({ headers }: { headers: Record<string,string> }) {
  const [query,setQuery] = useState('')
  const [roles,setRoles] = useState<Role[]>([])
  const [selected,setSelected] = useState('')
  const [report,setReport] = useState<Report | null>(null)
  const [busy,setBusy] = useState(false)
  const [error,setError] = useState('')
  const request = async (url: string) => {
    const response = await fetch(url,{ headers, signal:AbortSignal.timeout(60000) })
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || 'Research explorer unavailable')
    return data
  }
  const search = async () => {
    setBusy(true);setError('');setReport(null);setSelected('')
    try { const data = await request(`/api/career-roles?query=${encodeURIComponent(query)}`);setRoles(data.roles);if (!data.roles.length) setError('No matching occupation in the local catalog.') }
    catch (e) { setError(e instanceof Error ? e.message : 'Search failed') }
    finally { setBusy(false) }
  }
  const explore = async () => {
    setBusy(true);setError('')
    try { setReport(await request(`/api/career-explore?code=${encodeURIComponent(selected)}`)) }
    catch (e) { setError(e instanceof Error ? e.message : 'Exploration failed') }
    finally { setBusy(false) }
  }
  return <details className="border border-neutral-200 rounded-lg p-5 text-sm">
    <summary className="cursor-pointer font-medium">Historical Career Transition Experiment</summary>
    <div className="space-y-4 mt-4">
      <p className="text-neutral-500">Choose your current occupation explicitly. This research model uses historical role transitions, without your skills. Evidence-linked career ideas from Gemma remain separate. Searches and inference use local files.</p>
      <label className="block">Current occupation search<input aria-label="Current occupation search" value={query} maxLength={100} onChange={e=>setQuery(e.target.value)} className="block border border-neutral-200 rounded p-3 mt-2 w-full" placeholder="For example: accountant, teacher, software" /></label>
      <button disabled={busy || query.trim().length<2} onClick={search} className="border border-neutral-200 px-4 py-2 rounded disabled:opacity-40">Find occupations</button>
      {roles.length>0 && <><label className="block">Confirm your current occupation<select aria-label="Current occupation" value={selected} onChange={e=>{setSelected(e.target.value);setReport(null)}} className="block border border-neutral-200 p-3 rounded w-full mt-2"><option value="">Select an occupation</option>{roles.map(role=><option key={role.code} value={role.code}>{role.title}</option>)}</select></label><button onClick={explore} disabled={busy || !selected} className="bg-black text-white px-4 py-2 rounded disabled:opacity-40">Explore historical transitions</button></>}
      {busy && <p role="status">Loading local research results…</p>}
      {error && <p role="alert">{error}</p>}
      {report && <><p className="text-neutral-500">{report.available ? report.note : report.reason}</p>{report.roles?.map(role=><p key={role.code}>{role.title}<span className="block text-xs text-neutral-500">Model activation {role.activation?.toFixed(3)} · ESCO {role.code}</span></p>)}</>}
    </div>
  </details>
}
