'use client'

import Link from 'next/link'
import { useState, useRef } from 'react'
import examples from '../../data/review_examples.json'

type Decision = { review_status: string; reviewer: string; review_note: string; review_type: string }
const pending: Decision = { review_status: 'pending', reviewer: '', review_note: '', review_type: 'human' }
const labels = ['Vague', 'Biased', 'Experience_Inflated'] as const

export default function ReviewWorkspace() {
  const [rewrites, setRewrites] = useState(() => examples.rewrites.map(row => ({ ...row })))
  const [jobs, setJobs] = useState(() => examples.jobs.map(row => ({ ...row })))
  const [task, setTask] = useState<'rewrites' | 'jobs'>('rewrites')
  const [index, setIndex] = useState(0)
  const [reviewer, setReviewer] = useState('')
  const [message, setMessage] = useState('')
  const importInput = useRef<HTMLInputElement>(null)
  const rows = task === 'rewrites' ? rewrites : jobs
  const current = rows[index]
  const reviewed = rows.filter(row => row.review_status !== 'pending').length
  const rewrite = rewrites[index]
  const job = jobs[index]

  const decide = (status: 'approved' | 'rejected') => {
    if (!reviewer.trim()) { setMessage('Enter your name to record a review decision.'); return }
    if (task === 'rewrites') setRewrites(previous => previous.map((row, i) => i === index ? { ...row, review_status: status, reviewer: reviewer.trim(), review_type: 'human' } : row))
    else setJobs(previous => previous.map((row, i) => i === index ? { ...row, review_status: status, reviewer: reviewer.trim(), review_type: 'human' } : row))
    setMessage(`Recorded ${status}.`)
    if (index < rows.length-1) setIndex(index+1)
  }

  const download = () => {
    const content = JSON.stringify({ ...examples, rewrites, jobs }, null, 2)
    const url = URL.createObjectURL(new Blob([content], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url; anchor.download = 'careerlens-review-decisions.json'; anchor.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
    setMessage('Downloaded your review progress. Pending examples remain unavailable for training.')
  }

  const restore = async (file: File) => {
    try {
      if (file.size > 500000) throw new Error('Review progress files must be under 500 KB.')
      const saved = JSON.parse(await file.text())
      if (saved.version !== 1 || saved.rewrites?.length !== examples.rewrites.length || saved.jobs?.length !== examples.jobs.length) throw new Error('This file does not match the prepared review examples.')
      const decision = (row: Decision): Decision => {
        if (!['pending','approved','rejected'].includes(row.review_status) || typeof row.reviewer !== 'string') throw new Error('Invalid review decision.')
        return { review_status: row.reviewer.trim() ? row.review_status : 'pending', reviewer: row.reviewer.slice(0,100), review_note: String(row.review_note ?? '').slice(0,500), review_type: row.review_type === 'ai' ? 'ai' : 'human' }
      }
      const restoredRewrites = examples.rewrites.map((original, i) => {
        const row = saved.rewrites[i]
        if (row.source_id !== original.source_id || row.original !== original.original || typeof row.improved !== 'string' || row.improved.length > 1500) throw new Error('Original rewrite facts do not match.')
        return { ...original, improved: row.improved, ...decision(row) }
      })
      const restoredJobs = examples.jobs.map((original, i) => {
        const row = saved.jobs[i]
        if (row.source_id !== original.source_id || row.text !== original.text || labels.some(label => !['0','1'].includes(row[label]))) throw new Error('Job description facts or labels do not match.')
        return { ...original, ...Object.fromEntries(labels.map(label => [label,row[label]])), ...decision(row) }
      })
      setRewrites(restoredRewrites); setJobs(restoredJobs); setIndex(0); setMessage('Restored your downloaded review progress.')
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Could not read review progress.') }
  }

  return <main className="max-w-4xl mx-auto px-5 sm:px-8 py-10 space-y-7">
    <Link href="/" className="text-sm underline text-neutral-600">← Resume analysis</Link>
    <header className="space-y-3"><p className="text-xs uppercase tracking-widest text-neutral-500">CareerLens / Review workspace</p><h1 className="text-3xl font-medium">Review the training examples.</h1><p className="text-sm text-neutral-600">These fictional examples have been checked and approved by an AI reviewer for experimental training. You can inspect, correct, or replace those decisions with your own review. Changes stay in this tab until downloaded.</p><p className="text-xs text-neutral-500">Human approval can support an experimental training run. This small synthetic set cannot establish real-world accuracy or hiring validity.</p></header>
    <div className="flex flex-wrap gap-3 items-center justify-between"><div className="flex flex-wrap gap-2">{(['rewrites','jobs'] as const).map(name => <button key={name} onClick={() => { setTask(name); setIndex(0); setMessage('') }} className={`px-4 py-2 text-sm rounded-md border ${task === name ? 'bg-black text-white border-black' : 'border-neutral-200'}`}>{name === 'rewrites' ? `Resume rewrites (${rewrites.length})` : `Job descriptions (${jobs.length})`}</button>)}</div><div className="flex gap-4"><button onClick={() => importInput.current?.click()} className="text-sm underline">Restore progress</button><button onClick={download} className="text-sm underline">Download decisions</button></div><input aria-label="Restore review decisions" type="file" accept=".json,application/json" ref={importInput} className="hidden" onChange={e => { const file = e.target.files?.[0]; if (file) restore(file); e.target.value = '' }} /></div>
    <label className="block text-sm">Reviewer name<input value={reviewer} onChange={e => setReviewer(e.target.value)} maxLength={100} className="mt-2 block w-full sm:max-w-sm border border-neutral-200 p-3 rounded-md" autoComplete="name" placeholder="Your name for the review record" /></label>
    <div className="space-y-2"><p className="text-sm text-neutral-500">{reviewed} of {rows.length} reviewed · Example {index+1} · {current.review_type === 'ai' ? 'AI review' : 'Human review'} · {current.review_status.replaceAll('_',' ')}</p><progress aria-label="Review progress" value={reviewed} max={rows.length} className="w-full h-2 accent-black" /></div>
    {task === 'rewrites' ? <section className="space-y-4 border border-neutral-200 p-5 rounded-lg">
      <h2 className="font-medium">{rewrite.source_id}</h2><p className="text-sm text-neutral-500">{rewrite.jd_context}</p>
      <div className="grid sm:grid-cols-2 gap-5"><div><h3 className="text-xs uppercase tracking-wide text-neutral-500 mb-3">Original facts</h3><p className="text-sm bg-neutral-50 rounded-md p-4">{rewrite.original}</p></div><label className="text-sm"><span className="block text-xs uppercase tracking-wide text-neutral-500 mb-3">Draft target — edit if needed</span><textarea aria-label="Reviewed rewrite target" value={rewrite.improved} onChange={e => setRewrites(previous => previous.map((row, i) => i === index ? { ...row, improved: e.target.value, ...pending } : row))} className="w-full h-36 border border-neutral-200 rounded-md p-4" maxLength={1500} /></label></div>
      <p className="text-xs text-neutral-500">Verify identity, employers, numbers, technologies, negation, proficiency, and level of responsibility. Reject changes that add or remove facts.</p>
    </section> : <section className="space-y-5 border border-neutral-200 p-5 rounded-lg">
      <h2 className="font-medium">{job.source_id}</h2><p className="text-sm whitespace-pre-wrap bg-neutral-50 p-4 rounded-md">{job.text}</p>
      <p className="text-sm text-neutral-500">The checked values below are draft labels. Review each definition and correct the labels.</p>
      {labels.map(label => <label key={label} className="flex items-start gap-3 text-sm"><input type="checkbox" checked={job[label] === '1'} onChange={e => setJobs(previous => previous.map((row, i) => i === index ? { ...row, [label]: e.target.checked ? '1' : '0', ...pending } : row))} className="mt-1" /><span><span className="font-medium">{label.replaceAll('_',' ')}</span><span className="block text-neutral-500 mt-1">{examples.label_definitions[label]}</span></span></label>)}
    </section>}
    <div className="flex flex-wrap items-center justify-between gap-3"><button disabled={index === 0} onClick={() => setIndex(index-1)} className="text-sm border border-neutral-200 px-4 py-2 rounded-md disabled:opacity-40">Previous</button><div className="flex gap-3"><button onClick={() => decide('rejected')} className="text-sm border border-neutral-200 px-4 py-2 rounded-md">Reject example</button><button onClick={() => decide('approved')} className="text-sm bg-black text-white px-4 py-2 rounded-md">Approve example</button></div><button disabled={index === rows.length-1} onClick={() => setIndex(index+1)} className="text-sm border border-neutral-200 px-4 py-2 rounded-md disabled:opacity-40">Next</button></div>
    <p role="status" className="text-sm min-h-6 text-neutral-600">{message}</p>
    <details className="border-t border-neutral-200 pt-5 text-sm text-neutral-500"><summary className="cursor-pointer">Using your decisions</summary><p className="mt-3">Download the decisions before leaving this page. The project’s review-data tool imports that file and exports a training dataset only after every example in a task has an explicit decision. Reviewer names record your attestation; no identity verification is performed.</p></details>
  </main>
}
