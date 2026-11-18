'use client'

export type LocalReport = {
  available: boolean; model?: string; reason?: string; note?: string; career_note?: string
  alignment?: { job_quote: string; resume_quote: string; status: string; explanation: string }[]
  job_flags?: { phrase: string; category: string; explanation: string; suggestion: string }[]
  career_options?: { role: string; evidence: string[]; existing_skills: string[]; development: string[]; explanation: string }[]
  entities?: { name: string; kind: string; source: string; quote: string }[]
}

export default function LocalInsights({ report }: { report: LocalReport }) {
  if (!report.available) return <p className="text-sm border border-neutral-200 rounded-lg p-5">{report.reason}</p>
  return <div className="space-y-5">
    <p className="text-sm text-neutral-500">Local model: {report.model}. {report.note}</p>
    <section className="border border-neutral-200 rounded-lg p-5 space-y-4">
      <h3 className="font-medium">Contextual Qualification Matching</h3>
      {report.alignment?.length ? report.alignment.map((item, i) => <details key={i} className="border-t border-neutral-100 pt-3">
        <summary className="text-sm cursor-pointer">{item.status.replaceAll('_', ' ')} · {item.job_quote}</summary>
        <p className="text-sm mt-3">{item.explanation}</p>
        {item.resume_quote && <blockquote className="text-sm mt-3 pl-3 border-l-2 text-neutral-600">{item.resume_quote}</blockquote>}
      </details>) : <p className="text-sm text-neutral-500">No verified alignments returned. Use the qualification checklist above.</p>}
    </section>
    <section className="border border-neutral-200 rounded-lg p-5 space-y-4">
      <h3 className="font-medium">Contextual Job Wording Review</h3>
      {report.job_flags?.length ? report.job_flags.map((flag, i) => <div key={i} className="border-t border-neutral-100 pt-3 text-sm space-y-2">
        <p>“{flag.phrase}” · {flag.category}</p><p className="text-neutral-600">{flag.explanation}</p><p>Suggested wording: {flag.suggestion}</p>
      </div>) : <p className="text-sm text-neutral-500">The model returned no verified wording flags. This is not a fairness assessment.</p>}
      <p className="text-xs text-neutral-500">Use India Salary Comparison below for sourced pay references; this wording review does not estimate salary.</p>
    </section>
    <section className="border border-neutral-200 rounded-lg p-5 space-y-4">
      <h3 className="font-medium">Career Exploration</h3><p className="text-sm text-neutral-500">{report.career_note}</p>
      {report.career_options?.length ? report.career_options.map((option, i) => <details key={i} className="border-t border-neutral-100 pt-3">
        <summary className="text-sm cursor-pointer font-medium">{option.role}</summary><p className="text-sm mt-3">{option.explanation}</p>
        {option.evidence.map((quote, j) => <blockquote key={j} className="text-sm mt-3 pl-3 border-l-2 text-neutral-600">{quote}</blockquote>)}
        <p className="text-sm mt-3">Skills in your resume: {option.existing_skills.join(', ') || 'No canonical skills recognized'}</p>
        {option.development.length > 0 && <div className="mt-3 text-sm"><p>Suggested development:</p><ul className="list-disc pl-5 mt-2 space-y-1">{option.development.map((item, j) => <li key={j}>{item}</li>)}</ul></div>}
      </details>) : <p className="text-sm text-neutral-500">Insufficient verified evidence for role ideas.</p>}
    </section>
    {Boolean(report.entities?.length) && <details className="border border-neutral-200 rounded-lg p-5">
      <summary className="font-medium cursor-pointer">Model Extracted Entities and Sources</summary>
      <div className="mt-4 space-y-3">{report.entities?.map((entity, i) => <div key={i} className="text-sm border-t pt-3 border-neutral-100"><p>{entity.name} · {entity.kind} · {entity.source}</p><blockquote className="mt-2 pl-3 border-l-2 text-neutral-600">{entity.quote}</blockquote></div>)}</div>
    </details>}
  </div>
}
