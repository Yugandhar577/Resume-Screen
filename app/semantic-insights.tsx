'use client'
export type SemanticReport = {
  available: boolean; similarity?: number; reason?: string; note?: string
  attention?: Record<string, { quote: string; attention: number }[]>
  entity_graphs?: Record<string, {nodes:{name:string;kind:string;quote:string;start:number;end:number;method:string}[];relations:{source:number;target:number;kind:string}[]}>
  node_removal?: {name:string;quote:string;similarity_without_node:number;change:number}[]
  synthetic_calibration?: {synthetic_label_estimate:number;note:string;test_brier:number;test_ece:number}|null
}
export default function SemanticInsights({report}:{report:SemanticReport}) {
  return <section className="border border-neutral-200 rounded-lg p-5 space-y-3">
    <h3 className="font-medium">Experimental Semantic Similarity</h3><p className="text-sm">{report.available ? report.similarity : report.reason}</p><p className="text-xs text-neutral-500">{report.note}</p>
    {Object.entries(report.entity_graphs ?? {}).map(([name,graph])=><details key={name} className="text-sm"><summary className="cursor-pointer">{name} entity graph and observed relationships</summary><div className="mt-3 space-y-2">{graph.nodes.map((node,i)=><p key={i}>{i+1}. {node.kind} · {node.name}<span className="block text-xs text-neutral-500">“{node.quote}” · offsets {node.start}–{node.end} · {node.method}</span></p>)}{graph.relations.map((edge,i)=><p key={i} className="text-xs text-neutral-500">Node {edge.source+1} → {edge.target+1}: {edge.kind.replaceAll('_',' ')}</p>)}</div></details>)}
    {Object.entries(report.attention ?? {}).map(([name,chunks])=><details key={name}><summary className="text-sm cursor-pointer">{name} model attention (not causal attribution)</summary>{chunks.map((chunk,i)=><p key={i} className="text-sm mt-2">{chunk.attention.toFixed(3)} · {chunk.quote}</p>)}</details>)}
    {Boolean(report.node_removal?.length) && <details><summary className="text-sm cursor-pointer">Tested entity removal sensitivity</summary><p className="text-xs text-neutral-500 mt-3">The same trained model was rerun with one entity node removed, preserving source text nodes. A positive change means its similarity decreased. This does not predict a real-world effect.</p>{report.node_removal?.map((item,i)=><p key={i} className="text-sm mt-2">{item.name}: similarity without node {item.similarity_without_node.toFixed(4)}; change {item.change.toFixed(4)}</p>)}</details>}
    {report.synthetic_calibration && <details className="text-sm"><summary className="cursor-pointer">Synthetic benchmark calibration</summary><p className="mt-3 text-neutral-500">{report.synthetic_calibration.note}</p><p className="mt-2">Synthetic label estimate: {report.synthetic_calibration.synthetic_label_estimate.toFixed(3)} · Held-out Brier: {report.synthetic_calibration.test_brier.toFixed(4)} · Calibration error: {report.synthetic_calibration.test_ece.toFixed(4)}</p></details>}
  </section>
}
