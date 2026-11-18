'use client'

export type JDExperiment = {
  available: boolean; reason?: string; note?: string; version?: string
  labels?: { category: string; activation: number; flagged: boolean; sentence_removal: { quote: string; activation_drop: number }[] }[]
  weak_supervision?: { available:boolean; reason?:string; note?:string; labels?:{category:string;votes:number[];weak_probability:number|null}[] }
}

export default function ResearchInsights({ report }: { report: JDExperiment }) {
  return <section className="border border-neutral-200 rounded-lg p-5 space-y-4">
    <h3 className="font-medium">Experimental BERT Wording Analysis</h3>
    <p className="text-sm text-neutral-500">{report.available ? report.note : report.reason}</p>
    {report.labels?.map(item => <details key={item.category} className="border-t border-neutral-100 pt-3 text-sm">
      <summary className="cursor-pointer">{item.category.replaceAll('_',' ')} · {item.flagged ? 'Flagged by experimental model' : 'Below experimental threshold'}</summary>
      <p className="mt-3 text-neutral-500">Raw activation: {item.activation.toFixed(3)}. This is not a fairness probability.</p>
      {item.sentence_removal.map((evidence,i) => <blockquote key={i} className="mt-3 pl-3 border-l-2"><p>{evidence.quote}</p><p className="text-xs text-neutral-500 mt-1">Activation change when removed: {evidence.activation_drop.toFixed(4)}</p></blockquote>)}
    </details>)}
    {report.weak_supervision && <details className="text-sm border-t border-neutral-100 pt-3"><summary className="cursor-pointer">Experimental weak supervision comparison</summary><p className="mt-3 text-neutral-500">{report.weak_supervision.note || report.weak_supervision.reason}</p>{report.weak_supervision.labels?.map(item=><p key={item.category} className="mt-2">{item.category.replaceAll('_',' ')} · {item.weak_probability===null?'Unlabeled':`Weak probability ${item.weak_probability.toFixed(3)}`} · votes {item.votes.join(', ')}</p>)}</details>}
  </section>
}
