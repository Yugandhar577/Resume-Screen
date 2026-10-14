'use client'
import { useState } from 'react'

type Job = { id: string; title: string; company: string; description: string; url: string; location: string; source: string }

export default function JobBoard({ onSelect, accessToken = '' }: { onSelect: (description: string) => void; accessToken?: string }) {
  const [query, setQuery] = useState('')
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [searched, setSearched] = useState(false)
  const search = async () => {
    setLoading(true); setError('')
    try {
      const response = await fetch(`/api/jobs?query=${encodeURIComponent(query)}&limit=10`, { headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {}, signal: AbortSignal.timeout(20000) })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail ?? 'Job search unavailable')
      setJobs(data.jobs); setSearched(true)
    } catch (e) { setError(e instanceof Error ? e.message : 'Job search unavailable') }
    finally { setLoading(false) }
  }
  return <details className="border-t border-neutral-200 pt-4 text-sm">
    <summary className="cursor-pointer font-medium">Find a job description</summary>
    <p className="text-xs text-neutral-500 mt-3">Public listings from <a className="underline" href="https://www.arbeitnow.com" target="_blank" rel="noreferrer">Arbeitnow</a>, primarily Europe. Your resume is not sent to the job board.</p>
    <div className="flex gap-2 mt-3"><input aria-label="Job search keywords" value={query} maxLength={100} onChange={e => setQuery(e.target.value)} className="min-w-0 flex-1 border border-neutral-200 rounded-md p-2" placeholder="Role or skill" /><button disabled={loading} onClick={search} className="bg-black text-white px-4 rounded-md disabled:opacity-40">{loading ? 'Searching...' : 'Search'}</button></div>
    {error && <p role="alert" className="mt-3">{error}</p>}
    {searched && !jobs.length && <p className="text-neutral-500 mt-3">No matches in the cached feed. Try a broader keyword or paste a description.</p>}
    {jobs.map(job => <div key={job.id} className="mt-4 pt-4 border-t border-neutral-100"><p className="font-medium">{job.title}</p><p className="text-xs text-neutral-500">{job.company} · {job.location}</p><div className="flex gap-4 mt-2"><button className="underline" onClick={() => onSelect(job.description)}>Use description</button><a className="underline" href={job.url} target="_blank" rel="noreferrer">View on {job.source}</a></div></div>)}
  </details>
}
