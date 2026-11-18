'use client'
import { useState, useRef, useEffect } from 'react'
import JobBoard from './job-board'
import CareerExplorer from './career-explorer'
import MarketTools from './market-tools'
import EvidenceMatching, {type EvidenceReport} from './evidence-matching'
import SemanticInsights, { type SemanticReport } from './semantic-insights'
import ResearchInsights, { type JDExperiment } from './research-insights'
import LocalInsights, { type LocalReport } from './local-insights'
import Link from 'next/link'
import { Upload, FileText, Target, Download, Brain, BarChart3, Zap, TrendingUp, Shield, CheckCircle, AlertCircle, Activity, Network, Database, Play, Briefcase, Code, Globe, Sparkles, ChevronDown } from 'lucide-react'

type SystemStatus = { status: string; analysis_mode?: string; ml_engine: boolean; local_ai?: { available: boolean; model: string; reason?: string }; semantic_model?: { available: boolean; experimental: boolean; reason?: string }; capabilities?: { ocr: boolean } } | null
type Analysis = { atsScore: number | null; atsFeedback: string; techFeedback: string; hrFeedback: string; skillGaps: string[]; recommendations: string[] } | null
type GraphNode = { id: string; type: string; weight: number }
type GraphEdge = { source: string; target: string; weight: number }
type GraphData = { resume_graph: { nodes: GraphNode[]; edges: GraphEdge[] }; jd_graph: { nodes: GraphNode[]; edges: GraphEdge[] } } | null
type DetailedResults = {
  matching?: { graph_similarity: number; skill_overlap: string[]; missing_skills: string[]; match_score: number | null }
  bias_detection?: { bias_detected: boolean; bias_types: string[]; flags: { type: string; phrase: string }[]; note: string }
  graph_analysis?: NonNullable<GraphData>
  skill_evidence?: Record<string, string[]>
  requirements?: { items: Requirement[]; note: string }
  reviewers?: Reviewer[]
  local_review?: LocalReport
  evidence_matching?: EvidenceReport
  jd_experiment?: JDExperiment
  score_explanation?: { method: string; note: string; items: { skill: string; score_without_skill: number; point_change: number }[] }
  semantic_matching?: SemanticReport
} | null
type EditProposal = { id: string; line: number; original: string; proposed: string; evidence: string; reason: string }
type Requirement = { kind: string; requirement: string; priority: string; status: string; evidence: string[]; note: string; job_quote: string }
type Reviewer = { persona: string; method: string; rubric: string; strengths: { statement: string; evidence: string[] }[]; concerns: { statement: string; evidence: string[] }[]; recommendations: string[] }
type Optimization = { method: string; changes: string[]; note: string; sourceResume?: string; proposals?: EditProposal[]; templates?: { id: string; label: string }[]; applied_edits?: EditProposal[]; proposalToken?: string } | null

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a'); link.href = url; link.download = filename; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function GraphViz({ nodes, edges, color }: { nodes: GraphNode[]; edges: GraphEdge[]; color: string }) {
  const W = 400, H = 350
  if (!nodes.length) return <div className="flex items-center justify-center h-40 text-neutral-500 text-sm">No graph data</div>
  const list = nodes.slice(0, 12)
  const positioned = list.map((n, i, arr) => {
    const angle = (2 * Math.PI * i) / arr.length - Math.PI / 2
    const r = arr.length <= 1 ? 0 : 138
    return { ...n, x: W / 2 + r * Math.cos(angle), y: H / 2 + r * Math.sin(angle) }
  })
  const posMap: Record<string, { x: number; y: number }> = {}
  positioned.forEach(n => { posMap[n.id] = { x: n.x, y: n.y } })
  const sc: Record<string, string> = { skill: color, role: '#525252', company: '#a3a3a3' }
  const visibleEdges = edges.filter(e => posMap[e.source] && posMap[e.target]).slice(0, 20)
  return (
    <div className="entity-graph">
    <svg viewBox={`0 0 ${W} ${H}`} className="entity-graph-canvas" role="img" aria-label="Entity connections. Numbered nodes correspond to the full names listed below.">
      {visibleEdges.map((e, i) => {
        const s = posMap[e.source], t = posMap[e.target]
        if (!s || !t) return null
        return <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke="#d4d4d4" strokeWidth={1.5} />
      })}
      {positioned.map((n, i) => (
        <g key={i}>
          <circle cx={n.x} cy={n.y} r={21} fill="white" stroke={sc[n.type] ?? '#d4d4d4'} strokeWidth={2} />
          <text x={n.x} y={n.y+1} textAnchor="middle" dominantBaseline="middle" fontSize={16} fontWeight="600" fill="#171717">{i + 1}</text>
          <title>{i + 1}. {n.id} ({n.type})</title>
        </g>
      ))}
    </svg>
    <ol className="entity-graph-key" aria-label="Graph node names">
      {list.map((n, i) => <li key={n.id} className="entity-graph-key-item">
        <span className="entity-graph-number" style={{ borderColor: sc[n.type] ?? '#d4d4d4' }} aria-hidden="true">{i + 1}</span>
        <span className="entity-graph-name">{n.id}<span className="entity-graph-type">{n.type}</span></span>
      </li>)}
    </ol>
    {(nodes.length > list.length || edges.length > visibleEdges.length) && <p className="text-xs text-neutral-500">Showing {list.length} of {nodes.length} nodes and {visibleEdges.length} of {edges.length} connections.</p>}
    </div>
  )
}

const SAMPLE_DATASETS = {
  'software_engineer': {
    resume: `John Smith - Senior Software Engineer
john.smith@email.com | github.com/johnsmith | linkedin.com/in/johnsmith

EXPERIENCE
Senior Software Engineer @ TechCorp (2021-Present)
- Built ML pipelines with Python, PyTorch, TensorFlow serving 1M+ predictions/day
- Deployed microservices on AWS using Docker, Kubernetes, and Terraform
- Led team of 5 engineers, improved system latency by 40% and reduced costs by 30%
- Implemented CI/CD pipelines with Jenkins and GitHub Actions

Software Engineer @ StartupXYZ (2019-2021)
- Developed REST APIs with FastAPI and Node.js handling 500K requests/day
- Built recommendation system using collaborative filtering and neural networks
- Worked with PostgreSQL, MongoDB, Redis for data storage and caching

SKILLS
Languages: Python, JavaScript, TypeScript, SQL, Bash
Frameworks: React, Node.js, FastAPI, Django, Flask
ML/AI: PyTorch, TensorFlow, scikit-learn, Pandas, NumPy
Cloud: AWS (EC2, S3, Lambda, RDS), Docker, Kubernetes
Tools: Git, CI/CD, Jenkins, PostgreSQL, MongoDB, Redis

EDUCATION
B.S. Computer Science - University of Technology (2019)
GPA: 3.8/4.0`,
    jd: `Senior ML Engineer - AI Research Team

We are seeking a Senior ML Engineer to join our AI research team and drive the development of next-generation machine learning systems.

REQUIREMENTS
- 5+ years experience in machine learning and deep learning
- Expert-level Python programming with PyTorch and TensorFlow
- Experience with Graph Neural Networks (GNNs) and attention mechanisms
- Strong background in NLP, computer vision, or recommendation systems
- Experience with distributed training and model optimization
- Knowledge of MLOps, model deployment, and monitoring
- Proficiency with Docker, Kubernetes, and cloud platforms (AWS/GCP)
- Experience with large-scale data processing (Spark, Hadoop)

PREFERRED
- PhD in Computer Science, Machine Learning, or related field
- Publications in top-tier ML conferences (NeurIPS, ICML, ICLR)
- Experience with transformer architectures and attention mechanisms
- Knowledge of reinforcement learning and multi-agent systems
- Contributions to open-source ML frameworks

RESPONSIBILITIES
- Design and implement state-of-the-art ML algorithms
- Lead research initiatives and mentor junior engineers
- Collaborate with product teams to deploy ML solutions at scale
- Publish research findings and represent company at conferences

We offer competitive compensation, equity, and the opportunity to work on cutting-edge AI research with world-class team.`
  },
  'data_scientist': {
    resume: `Sarah Johnson - Data Scientist
sarah.j@email.com | Portfolio: sarahjohnson.com

EXPERIENCE
Data Scientist @ DataCorp (2020-Present)
- Built predictive models using XGBoost, Random Forest achieving 92% accuracy
- Performed A/B testing and statistical analysis for product features
- Created dashboards with Tableau and Power BI for executive reporting
- Analyzed customer behavior data using SQL, Python, and R

Junior Data Analyst @ Analytics Inc (2018-2020)
- Cleaned and processed large datasets using Pandas and SQL
- Built regression models for sales forecasting
- Automated reporting workflows with Python scripts

SKILLS
Python, R, SQL, Excel, Tableau, Power BI
Machine Learning, Statistics, A/B Testing
Pandas, NumPy, scikit-learn, XGBoost
PostgreSQL, MySQL, BigQuery

EDUCATION
M.S. Data Science - State University (2018)
B.S. Statistics - State University (2016)`,
    jd: `Senior Data Scientist - Product Analytics

REQUIREMENTS
- 4+ years experience in data science and analytics
- Strong Python and SQL skills
- Experience with statistical modeling and A/B testing
- Machine learning experience (classification, regression, clustering)
- Data visualization with Tableau or similar tools
- Experience with cloud data warehouses (BigQuery, Snowflake, Redshift)

PREFERRED
- Experience with causal inference and experimentation
- Knowledge of deep learning frameworks
- Product analytics experience

RESPONSIBILITIES
- Build predictive models to drive product decisions
- Design and analyze A/B tests
- Create dashboards and reports for stakeholders
- Collaborate with product and engineering teams`
  },
  'devops_engineer': {
    resume: `Mike Chen - DevOps Engineer
mike.chen@email.com | github.com/mikechen

EXPERIENCE
DevOps Engineer @ CloudTech (2020-Present)
- Managed Kubernetes clusters serving 100+ microservices
- Implemented Infrastructure as Code using Terraform and Ansible
- Built CI/CD pipelines with Jenkins, GitLab CI, and ArgoCD
- Reduced deployment time by 60% through automation
- Monitored systems with Prometheus, Grafana, and ELK stack

Systems Administrator @ HostingCo (2018-2020)
- Managed Linux servers and networking infrastructure
- Automated tasks with Bash and Python scripts
- Implemented backup and disaster recovery procedures

SKILLS
Cloud: AWS, GCP, Azure
Containers: Docker, Kubernetes, Helm
IaC: Terraform, Ansible, CloudFormation
CI/CD: Jenkins, GitLab CI, GitHub Actions, ArgoCD
Monitoring: Prometheus, Grafana, ELK, Datadog
Languages: Python, Bash, Go

EDUCATION
B.S. Information Technology (2018)`,
    jd: `Senior DevOps Engineer - Platform Team

REQUIREMENTS
- 5+ years DevOps/SRE experience
- Expert with Kubernetes and Docker
- Strong Infrastructure as Code skills (Terraform, Ansible)
- CI/CD pipeline design and implementation
- Cloud platforms (AWS, GCP, or Azure)
- Monitoring and observability (Prometheus, Grafana)
- Scripting with Python, Bash, or Go

PREFERRED
- Experience with service mesh (Istio, Linkerd)
- GitOps workflows (ArgoCD, Flux)
- Security best practices and compliance

RESPONSIBILITIES
- Design and maintain Kubernetes infrastructure
- Build automated deployment pipelines
- Implement monitoring and alerting systems
- Optimize cloud costs and performance
- Mentor junior engineers`
  },
}

export default function CareerLens() {
  const [resume, setResume] = useState('')
  const [jobDescription, setJobDescription] = useState('')
  const [analyzedJob, setAnalyzedJob] = useState('')
  const [analysis, setAnalysis] = useState<Analysis>(null)
  const [optimizedResume, setOptimizedResume] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploadMode, setUploadMode] = useState<'text' | 'pdf'>('text')
  const [graphData, setGraphData] = useState<GraphData>(null)
  const [systemStatus, setSystemStatus] = useState<SystemStatus>(null)
  const [detailedResults, setDetailedResults] = useState<DetailedResults>(null)
  const [activeTab, setActiveTab] = useState<'input' | 'results' | 'graphs' | 'optimize'>('input')
  const [selectedSample, setSelectedSample] = useState('')
  const [showSampleMenu, setShowSampleMenu] = useState(false)
  const [error, setError] = useState('')
  const [warnings, setWarnings] = useState<string[]>([])
  const [optimization, setOptimization] = useState<Optimization>(null)
  const [reviewMethod, setReviewMethod] = useState('Rule_Based')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [useAI, setUseAI] = useState(false)
  const [accessToken, setAccessToken] = useState('')
  const authHeaders: Record<string, string> = accessToken ? { Authorization: `Bearer ${accessToken}` } : {}
  const [useSemantic, setUseSemantic] = useState(false)
  const [useResearch, setUseResearch] = useState(false)
  const [useEvidence, setUseEvidence] = useState(true)
  const [rewriteEngine, setRewriteEngine] = useState('gemma')
  const [useOCR, setUseOCR] = useState(false)
  const [pdfPassword, setPdfPassword] = useState('')
  const [acceptedEdits, setAcceptedEdits] = useState<string[]>([])
  const [template, setTemplate] = useState('original')
  const [editLoading, setEditLoading] = useState(false)
  const [proposalLoading, setProposalLoading] = useState(false)
  const [sourceDraft, setSourceDraft] = useState('')
  const [proposalNote, setProposalNote] = useState('')
  const [comparison, setComparison] = useState<{ before: number | null; after: number | null; delta: number | null } | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { checkStatus() }, [])
  const checkStatus = async () => {
    try {
      const response = await fetch('/health', { signal: AbortSignal.timeout(10000) })
      if (!response.ok) throw new Error('Backend unavailable')
      const status = await response.json()
      setSystemStatus(status)
    }
    catch { setSystemStatus({ status: 'offline', ml_engine: false }) }
  }

  const loadSample = (key: string) => {
    const s = SAMPLE_DATASETS[key as keyof typeof SAMPLE_DATASETS]
    if (s) { setResume(s.resume); setJobDescription(s.jd); setSelectedSample(key); setShowSampleMenu(false); setUploadMode('text'); setSelectedFile(null) }
  }

  const runAnalysis = async () => {
    if (uploadMode === 'pdf') { if (selectedFile && jobDescription.trim()) await runPDF(selectedFile); return }
    if (!resume.trim() || !jobDescription.trim()) return
    resetResults()
    setAnalyzedJob(jobDescription)
    try {
      const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders }, body: JSON.stringify({ resume, jobDescription, useAI, useSemantic, useResearch, useEvidence }), signal: AbortSignal.timeout(330000) })
      await applyResponse(response)
    } catch (e) { setError(e instanceof Error ? e.message : 'Analysis failed') } finally { setLoading(false) }
  }

  const runPDF = async (file: File) => {
    resetResults()
    setAnalyzedJob(jobDescription)
    try {
      const form = new FormData(); form.append('file', file); form.append('jobDescription', jobDescription)
      form.append('useAI', String(useAI))
      form.append('useSemantic', String(useSemantic))
      form.append('useResearch', String(useResearch))
      form.append('useEvidence',String(useEvidence))
      form.append('useOCR', String(useOCR)); form.append('pdfPassword', pdfPassword)
      const response = await fetch('/api/analyze-pdf', { method: 'POST', headers: authHeaders, body: form, signal: AbortSignal.timeout(390000) })
      await applyResponse(response)
    } catch (e) { setError(e instanceof Error ? e.message : 'PDF analysis failed') } finally { setLoading(false) }
  }

  const resetResults = () => {
    setLoading(true); setError(''); setWarnings([]); setAnalysis(null); setOptimizedResume('');
    setDetailedResults(null); setGraphData(null); setOptimization(null); setActiveTab('results')
    setAcceptedEdits([]); setTemplate('original'); setComparison(null)
    setProposalNote(''); setSourceDraft('')
  }

  const applyResponse = async (response: Response) => {
    const data = await response.json().catch(() => null)
    if (!response.ok) throw new Error(typeof data?.detail === 'string' ? data.detail : `Analysis failed (${response.status}). Check your inputs and backend connection.`)
    if (!data?.analysis || typeof data.optimizedResume !== 'string') throw new Error('The backend returned an invalid result.')
    setAnalysis(data.analysis); setOptimizedResume(data.optimizedResume); setDetailedResults(data.detailed_results ?? null)
    setGraphData(data.detailed_results?.graph_analysis ?? null); setOptimization(data.optimization ?? null)
    setWarnings(data.warnings ?? []); setReviewMethod(data.review_method ?? 'Rule_Based')
    setComparison(data.comparison ?? null)
    setSourceDraft(data.optimization?.sourceResume ?? '')
  }

  const previewEdits = async () => {
    if (!optimization?.sourceResume) return
    setEditLoading(true); setError('')
    try {
      const response = await fetch('/api/optimize-preview', { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({ resume: optimization.sourceResume, jobDescription: analyzedJob, acceptedEdits, template, proposalToken: optimization.proposalToken ?? '' }), signal: AbortSignal.timeout(60000) })
      await applyResponse(response)
    } catch (e) { setError(e instanceof Error ? e.message : 'Edit preview failed') }
    finally { setEditLoading(false) }
  }

  const generateLocalEdits = async () => {
    if (!optimization?.sourceResume) return
    setProposalLoading(true); setError(''); setProposalNote('')
    try {
      const response = await fetch('/api/rewrite-proposals', { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({ resume: optimization.sourceResume, jobDescription: analyzedJob, engine: rewriteEngine }), signal: AbortSignal.timeout(630000) })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Local proposals unavailable')
      setOptimization(current => current ? { ...current, proposals: data.proposals, proposalToken: data.proposalToken } : current)
      setAcceptedEdits([]); setProposalNote(`${data.note} Rejected proposals: ${data.rejected}.`)
    } catch (e) { setError(e instanceof Error ? e.message : 'Local proposals unavailable') }
    finally { setProposalLoading(false) }
  }

  const reanalyzeSource = async () => {
    if (!sourceDraft.trim()) return
    const corrected = sourceDraft
    resetResults()
    try {
      const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({ resume: corrected, jobDescription: analyzedJob, useAI, useSemantic, useResearch, useEvidence }), signal: AbortSignal.timeout(330000) })
      await applyResponse(response)
      setResume(corrected); setUploadMode('text'); setSelectedFile(null)
    } catch (e) { setError(e instanceof Error ? e.message : 'Corrected text analysis failed') }
    finally { setLoading(false) }
  }

  const downloadAsPDF = async () => {
    if (!optimizedResume) return
    setError('')
    try {
      const response = await fetch('/api/export-pdf', { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({ resume: optimizedResume }), signal: AbortSignal.timeout(60000) })
      if (!response.ok) {
        const data = await response.json().catch(() => null)
        throw new Error(typeof data?.detail === 'string' ? data.detail : 'PDF export failed. Check the local backend connection.')
      }
      downloadBlob(await response.blob(), 'careerlens-resume.pdf')
    } catch (e) { setError(e instanceof Error ? e.message : 'PDF export failed') }
  }

  const isOnline = systemStatus?.status === 'healthy'
  const sampleLabels: Record<string, string> = { software_engineer: 'Software Engineer', data_scientist: 'Data Scientist', devops_engineer: 'DevOps Engineer' }
  const tabs = [
    { id: 'input' as const, label: 'Analyze', icon: <FileText className="w-3.5 h-3.5" /> },
    { id: 'results' as const, label: 'Results', icon: <BarChart3 className="w-3.5 h-3.5" /> },
    { id: 'graphs' as const, label: 'Graphs', icon: <Network className="w-3.5 h-3.5" /> },
    { id: 'optimize' as const, label: 'Resume', icon: <Sparkles className="w-3.5 h-3.5" /> },
  ]
  return (
    <>
    <article className="print-resume" aria-hidden="true">{optimizedResume}</article>
    <div className="min-h-screen bg-white">
      <header className="bg-white border-b border-neutral-200 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-5 sm:px-8 h-20 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-md bg-neutral-900 flex items-center justify-center"><Brain className="w-4 h-4 text-white" /></div>
            <span className="font-bold text-lg tracking-tight text-neutral-900">CareerLens</span>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            {tabs.map(t => (<button key={t.id} onClick={() => setActiveTab(t.id)} className={`flex items-center gap-1.5 px-1 py-2 border-b-2 text-sm font-medium transition-colors ${activeTab === t.id ? 'border-black text-black' : 'border-transparent text-neutral-500 hover:text-black'}`}>{t.icon}{t.label}</button>))}
          </nav>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-5 sm:px-8 py-10 sm:py-14">
        <nav aria-label="Sections" className="md:hidden flex flex-wrap gap-2 mb-6">{tabs.map(t => <button key={t.id} onClick={() => setActiveTab(t.id)} className={`px-3 py-2 rounded-lg text-sm ${activeTab === t.id ? 'bg-neutral-900 text-white' : 'bg-white text-neutral-600'}`}>{t.label}</button>)}</nav>
        {error && <div role="alert" className="mb-6 p-4 rounded-md border border-black bg-neutral-50 text-neutral-900"><span className="font-semibold">Unable to complete this request. </span>{error}</div>}
        {warnings.map((warning, i) => <p key={i} className="mb-3 p-3 rounded-md bg-neutral-50 text-neutral-900 text-sm"><span className="font-medium">Note: </span>{warning}</p>)}

        {activeTab === 'input' && (
          <div className="space-y-8">
            <details className="text-xs text-neutral-500"><summary className="cursor-pointer">Private deployment access</summary><label className="block mt-2">Access token (only if required)<input type="password" autoComplete="off" value={accessToken} onChange={e => setAccessToken(e.target.value)} className="block border border-neutral-200 rounded-md p-2 mt-1 w-full max-w-md" /></label><p className="mt-2">Kept in memory for this page only. Use HTTPS outside local development.</p></details>
            <div className="max-w-3xl pt-2 pb-2 sm:pb-5">
              <p className="text-xs uppercase tracking-[0.18em] text-neutral-500 mb-5">Resume / Role fit</p>
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-medium tracking-tight leading-[1.12] text-black mb-5">Make your experience clear.</h1>
              <p className="text-neutral-600 text-base sm:text-lg leading-relaxed max-w-xl">Compare your resume with a job, review skill evidence, and export a clean copy.</p>
            </div>
            <div className="flex flex-wrap gap-x-6 gap-y-3">
              {[{ icon: <Brain className="w-3.5 h-3.5" />, label: 'Skill Coverage', on: isOnline }, { icon: <Zap className="w-3.5 h-3.5" />, label: 'Local Gemma', on: systemStatus?.local_ai?.available }, { icon: <Network className="w-3.5 h-3.5" />, label: 'Entity Graphs', on: isOnline }, { icon: <Shield className="w-3.5 h-3.5" />, label: 'JD Phrase Screening', on: isOnline }].map(({ icon, label, on }, i) => (<div key={i} className={`flex items-center gap-1.5 text-xs ${on ? 'text-neutral-800' : 'text-neutral-500'}`}>{icon}{label}</div>))}
            </div>

            {/* Sample Dataset Selector */}
            <div className="flex">
              <div className="relative">
                <button onClick={() => setShowSampleMenu(!showSampleMenu)} className="flex items-center gap-2 bg-white border border-neutral-200 hover:border-neutral-400 text-neutral-700 text-sm font-medium px-5 py-2.5 rounded-md transition-colors">
                  <Play className="w-3.5 h-3.5 text-neutral-500" />
                  {selectedSample ? sampleLabels[selectedSample] : 'Try an example'}
                  <ChevronDown className="w-3.5 h-3.5 text-neutral-500" />
                </button>
                {showSampleMenu && (
                  <div className="absolute top-full mt-2 left-0 bg-white border border-neutral-200 rounded-md z-10 min-w-56 overflow-hidden">
                    {Object.entries(sampleLabels).map(([key, label]) => (
                      <button key={key} onClick={() => loadSample(key)} className="w-full text-left px-4 py-3 text-sm text-neutral-700 hover:bg-neutral-50 flex items-center gap-3 border-b border-neutral-50 last:border-0">
                        <div className="w-7 h-7 rounded-lg bg-neutral-100 flex items-center justify-center"><Briefcase className="w-3.5 h-3.5 text-neutral-500" /></div>
                        <div><p className="font-medium">{label}</p><p className="text-xs text-neutral-500">Resume + Job Description</p></div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="grid lg:grid-cols-2 gap-6">
              <div className="bg-white border border-neutral-200 rounded-lg p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-md bg-neutral-50 flex items-center justify-center"><FileText className="w-4 h-4 text-neutral-900" /></div>
                    <div><p className="font-semibold text-sm text-neutral-900">Resume</p><p className="text-neutral-500 text-xs">Paste text or upload PDF</p></div>
                  </div>
                  <div className="flex items-center gap-1 bg-neutral-100 rounded-lg p-1">
                    {(['text', 'pdf'] as const).map(m => (<button key={m} onClick={() => setUploadMode(m)} className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${uploadMode === m ? 'bg-white text-neutral-900' : 'text-neutral-500 hover:text-neutral-700'}`}>{m.toUpperCase()}</button>))}
                  </div>
                </div>
                {uploadMode === 'text' ? (
                  <textarea aria-label="Resume text" maxLength={50000} value={resume} onChange={e => setResume(e.target.value)} placeholder="Paste your resume here or load a sample above..." className="w-full h-72 bg-white border border-neutral-200 rounded-md p-4 text-sm text-neutral-800 placeholder-neutral-400 resize-none focus:outline-none focus:border-neutral-900 focus:ring-2 focus:ring-neutral-900/10 transition-colors font-mono" />
                ) : (
                  <div role="button" tabIndex={0} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click() }} onClick={() => fileInputRef.current?.click()} className="h-72 border-2 border-dashed border-neutral-200 rounded-md flex flex-col items-center justify-center gap-4 cursor-pointer hover:border-neutral-900 hover:bg-neutral-50 transition-colors group">
                    <div className="w-14 h-14 rounded-lg bg-neutral-100 flex items-center justify-center group-hover:bg-neutral-50 transition-colors"><Upload className="w-6 h-6 text-neutral-500 group-hover:text-neutral-900" /></div>
                    <div className="text-center"><p className="font-semibold text-sm text-neutral-700">{selectedFile ? selectedFile.name : 'Choose a PDF resume'}</p><p className="text-neutral-500 text-xs mt-1">click to browse — text PDFs up to 5 MB</p></div>
                    <input aria-label="PDF resume" ref={fileInputRef} type="file" accept=".pdf" onChange={e => {
                      const f = e.target.files?.[0]; setError('')
                      if (!f) return
                      if (!f.name.toLowerCase().endsWith('.pdf') || f.size > 5 * 1024 * 1024) { setError('Choose a PDF of 5 MB or smaller.'); setSelectedFile(null); return }
                      setSelectedFile(f)
                    }} className="hidden" />
                  </div>
                )}
                {uploadMode === 'pdf' && <div className="space-y-3 text-sm"><label className="flex gap-2"><input type="checkbox" disabled={!systemStatus?.capabilities?.ocr} checked={useOCR} onChange={e => setUseOCR(e.target.checked)} />Local OCR for scanned pages {systemStatus?.capabilities?.ocr ? '(review recognized text)' : '(not installed)'}</label><input aria-label="PDF password" type="password" autoComplete="off" maxLength={200} value={pdfPassword} onChange={e => setPdfPassword(e.target.value)} className="border border-neutral-200 rounded-md p-2 w-full" placeholder="PDF password, if needed (not stored)" /></div>}
              </div>
              <div className="bg-white border border-neutral-200 rounded-lg p-6 space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-md bg-neutral-50 flex items-center justify-center"><Target className="w-4 h-4 text-neutral-900" /></div>
                  <div><p className="font-semibold text-sm text-neutral-900">Job Description</p><p className="text-neutral-500 text-xs">Target role requirements</p></div>
                </div>
                <textarea aria-label="Job description" maxLength={50000} value={jobDescription} onChange={e => setJobDescription(e.target.value)} placeholder="Paste the job description here or load a sample above..." className="w-full h-72 bg-white border border-neutral-200 rounded-md p-4 text-sm text-neutral-800 placeholder-neutral-400 resize-none focus:outline-none focus:border-neutral-900 focus:ring-2 focus:ring-neutral-900/10 transition-colors font-mono" />
                <JobBoard accessToken={accessToken} onSelect={description => { setJobDescription(description); setSelectedSample('') }} />
              </div>
            </div>
            <label className="flex gap-3 text-sm text-neutral-600"><input type="checkbox" checked={useEvidence} onChange={e=>setUseEvidence(e.target.checked)}/><span>Find related resume passages for each job requirement with the local semantic encoder. Source quotes need qualification review.</span></label>
            <label className="flex gap-3 text-sm text-neutral-600"><input type="checkbox" checked={useResearch} onChange={e => setUseResearch(e.target.checked)} /><span>Include experimental BERT wording analysis. Trained on AI-reviewed fictional examples.</span></label>
            {systemStatus?.semantic_model?.available && <label className="flex gap-3 text-sm text-neutral-600"><input type="checkbox" checked={useSemantic} onChange={e => setUseSemantic(e.target.checked)} /><span>Include experimental local semantic matching. Evaluated on synthetic data only; the baseline score stays separate.</span></label>}
            <section className="border border-neutral-200 rounded-lg p-5 space-y-3 text-sm">
              <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="font-medium">Detailed AI review</h2><button onClick={() => checkStatus()} className="underline text-neutral-600">Refresh connection</button></div>
              <p>{systemStatus?.local_ai?.model ?? 'Gemma via Ollama'} · {systemStatus?.local_ai?.reason ?? 'Waiting for backend connection'}</p>
              <label className="flex items-start gap-3 text-neutral-600"><input type="checkbox" disabled={!systemStatus?.local_ai?.available} checked={useAI} onChange={e => setUseAI(e.target.checked)} className="mt-1 accent-black" /><span><span className="block font-medium text-neutral-900">Include detailed AI review</span><span className="block mt-1">Add ATS, technical, and recruiter feedback, contextual qualification review, and career ideas using Gemma on this computer. This may take a few minutes. Skill coverage and qualification checks run locally on every analysis.</span></span></label>
            </section>
            <div className="flex justify-center pt-4">
              <button onClick={runAnalysis} disabled={loading || (uploadMode === 'text' ? !resume.trim() : !selectedFile) || !jobDescription.trim()} className="flex items-center gap-3 bg-neutral-900 hover:bg-neutral-700 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium text-sm px-8 py-3.5 rounded-md transition-colors">
                {loading ? (<><div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /><span>Analyzing...</span></>) : (<><Brain className="w-5 h-5" /><span>Analyze Resume</span><Zap className="w-4 h-4" /></>)}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'results' && (
          <div className="space-y-6">
            {loading && (<div className="flex flex-col items-center justify-center py-32 gap-6"><div className="relative w-20 h-20"><div className="absolute inset-0 border-4 border-neutral-100 rounded-full" /><div className="absolute inset-0 border-4 border-neutral-900 border-t-transparent rounded-full animate-spin" /><Brain className="absolute inset-0 m-auto w-6 h-6 text-neutral-700" /></div><div className="text-center"><p className="font-bold text-lg text-neutral-900">Analyzing Resume</p><p className="text-neutral-500 text-sm mt-1">Checking skill coverage, resume structure, and job wording</p></div></div>)}
            {!loading && !analysis && (<div className="flex flex-col items-center justify-center py-32 gap-4 text-center"><div className="w-20 h-20 rounded-lg bg-neutral-100 flex items-center justify-center"><BarChart3 className="w-9 h-9 text-neutral-300" /></div><p className="font-semibold text-neutral-500 text-lg">No results yet</p><p className="text-neutral-500 text-sm">Run an analysis from the Analyze tab</p><button onClick={() => setActiveTab('input')} className="mt-2 px-6 py-2.5 bg-neutral-900 text-white text-sm font-medium rounded-md hover:bg-neutral-700 transition-colors">Go to Analyze</button></div>)}
            {!loading && analysis && (<>
              <div className="bg-white border border-neutral-200 rounded-lg p-8 flex flex-col md:flex-row items-center gap-8">
                <div className="relative w-40 h-40 shrink-0">
                  <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="42" stroke="#f5f5f5" strokeWidth="8" fill="none" />
                    <circle cx="50" cy="50" r="42" stroke="#171717" strokeWidth="8" fill="none" strokeLinecap="round" strokeDasharray={`${((analysis.atsScore ?? 0)/100)*263.9} 263.9`} />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center"><span className="text-4xl font-black text-neutral-900">{analysis.atsScore ?? 'N/A'}</span><span className="text-neutral-500 text-xs font-medium">/100</span></div>
                </div>
                <div className="flex-1 space-y-4">
                  <div><h2 className="text-2xl font-bold text-neutral-900">Recognized Skill Coverage</h2><p className="text-neutral-500 text-sm mt-1">{reviewMethod === 'Local_Ollama' ? 'Rule-based coverage with local AI reviewer perspectives' : 'Coverage uses evidence rules. Optional semantic results are separate.'}</p></div>
                  {detailedResults?.matching && (<div className="grid grid-cols-3 gap-3">{[{ label: 'Skill-set Similarity', value: `${(detailedResults.matching.graph_similarity*100).toFixed(0)}%` }, { label: 'Skill Matches', value: detailedResults.matching.skill_overlap?.length??0 }, { label: 'Skill Gaps', value: detailedResults.matching.missing_skills?.length??0 }].map(({ label, value }, i) => (<div key={i} className="bg-neutral-50 border border-neutral-100 rounded-md p-4 text-center"><div className="text-2xl font-bold text-neutral-900">{value}</div><div className="text-neutral-500 text-xs mt-1">{label}</div></div>))}</div>)}
                </div>
              </div>
              <div className="grid md:grid-cols-3 gap-4">{[{ title: 'ATS & Structure', content: analysis.atsFeedback, icon: <Network className="w-4 h-4" />, c: 'text-neutral-900', bg: 'bg-neutral-50', b: 'border-neutral-200' }, { title: 'Technical Evidence', content: analysis.techFeedback, icon: <Brain className="w-4 h-4" />, c: 'text-neutral-900', bg: 'bg-neutral-50', b: 'border-neutral-200' }, { title: 'Recruiter Review', content: analysis.hrFeedback, icon: <TrendingUp className="w-4 h-4" />, c: 'text-neutral-900', bg: 'bg-neutral-50', b: 'border-neutral-200' }].map(({ title, content, icon, c, bg, b }, i) => (<div key={i} className="bg-white border border-neutral-200 rounded-lg p-5 space-y-3"><div className={`flex items-center gap-2 text-sm font-semibold ${c}`}><div className={`w-7 h-7 rounded-lg ${bg} border ${b} flex items-center justify-center`}>{icon}</div>{title}</div><p className="text-neutral-500 text-sm leading-relaxed">{content}</p></div>))}</div>
              {(analysis.skillGaps?.length??0)>0 && (<div className="bg-white border border-neutral-200 rounded-lg p-5 space-y-3"><div className="flex items-center gap-2 text-sm font-semibold text-neutral-900"><AlertCircle className="w-4 h-4" />Skill Gaps ({analysis.skillGaps.length})</div><div className="flex flex-wrap gap-2">{analysis.skillGaps.map((s,i)=><span key={i} className="bg-neutral-50 border border-neutral-200 text-neutral-900 text-xs px-3 py-1.5 rounded-md font-medium">{s}</span>)}</div></div>)}
              {(analysis.recommendations?.length??0)>0 && (<div className="bg-white border border-neutral-200 rounded-lg p-5 space-y-3"><div className="flex items-center gap-2 text-sm font-semibold text-neutral-900"><CheckCircle className="w-4 h-4" />Recommendations</div><div className="space-y-2">{analysis.recommendations.map((r,i)=>(<div key={i} className="flex items-start gap-3 text-sm text-neutral-600"><div className="w-5 h-5 rounded-full bg-neutral-50 flex items-center justify-center shrink-0 mt-0.5 text-xs font-bold text-neutral-900">{i+1}</div>{r}</div>))}</div></div>)}
              {detailedResults?.bias_detection && (<div className="bg-white border border-neutral-200 rounded-lg p-5 space-y-3"><h3 className="font-semibold text-neutral-900">Job Wording Review</h3><p className="text-sm text-neutral-500">{detailedResults.bias_detection.note}</p>{detailedResults.bias_detection.flags.length ? detailedResults.bias_detection.flags.map((flag, i) => <p key={i} className="text-sm text-neutral-900">“{flag.phrase}” — {flag.type}</p>) : <p className="text-sm text-neutral-600">No phrases from the screening list were found.</p>}</div>)}
              {detailedResults?.matching?.skill_overlap.length ? <div className="bg-white border border-neutral-200 rounded-lg p-5"><h3 className="font-semibold text-sm mb-3">Resume Evidence for Matched Skills</h3>{detailedResults.matching.skill_overlap.map(skill => <details key={skill} className="py-2 border-b border-neutral-100"><summary className="cursor-pointer text-sm font-medium">{skill}</summary>{detailedResults.skill_evidence?.[skill]?.map((line, i) => <p key={i} className="mt-2 pl-3 border-l-2 text-sm text-neutral-600">{line}</p>)}</details>)}</div> : null}
              {detailedResults?.requirements && <section className="border border-neutral-200 rounded-lg p-5 space-y-4"><h3 className="font-medium">Qualification Evidence Checklist</h3><p className="text-sm text-neutral-500">{detailedResults.requirements.note}</p>{detailedResults.requirements.items.map((item, i) => <details key={i} className="border-t border-neutral-100 pt-3"><summary className="text-sm cursor-pointer">{item.requirement} · {item.priority} · {item.status.replaceAll('_', ' ')}</summary><p className="mt-2 text-xs text-neutral-500">Job: {item.job_quote}</p>{item.evidence.map((quote, j) => <p key={j} className="pl-3 border-l-2 mt-2 text-sm">{quote}</p>)}<p className="mt-2 text-xs text-neutral-500">{item.note}</p></details>)}</section>}
              {detailedResults?.reviewers?.map((reviewer, i) => <section key={i} className="border border-neutral-200 rounded-lg p-5 space-y-3"><h3 className="font-medium">{reviewer.persona}</h3><p className="text-xs text-neutral-500">{reviewer.method} · {reviewer.rubric}</p>{[...reviewer.strengths, ...reviewer.concerns].map((finding, j) => <div key={j}><p className="text-sm">{finding.statement}</p>{finding.evidence.map((quote, k) => <blockquote key={k} className="mt-2 pl-3 border-l-2 text-sm text-neutral-500">{quote}</blockquote>)}</div>)}{reviewer.recommendations.map((rec, j) => <p key={j} className="text-sm text-neutral-600">{rec}</p>)}</section>)}
              {detailedResults?.evidence_matching && <EvidenceMatching report={detailedResults.evidence_matching}/>}
              {detailedResults?.jd_experiment && <ResearchInsights report={detailedResults.jd_experiment} />}
              {detailedResults?.semantic_matching && <SemanticInsights report={detailedResults.semantic_matching} />}
              <MarketTools resume={optimization?.sourceResume || sourceDraft} jobDescription={analyzedJob} headers={authHeaders} />
              {useResearch && <CareerExplorer headers={authHeaders} />}
              {detailedResults?.local_review && <LocalInsights report={detailedResults.local_review} />}
              {optimization?.sourceResume && <details className="border border-neutral-200 rounded-lg p-5"><summary className="font-medium cursor-pointer">Review or Correct Extracted Resume Text</summary><p className="mt-3 text-sm text-neutral-500">Check reading order and OCR before using these results. Corrections start a fresh analysis and discard old proposals.</p><textarea aria-label="Extracted resume text" maxLength={50000} value={sourceDraft} onChange={e => setSourceDraft(e.target.value)} className="w-full h-64 mt-3 border border-neutral-200 p-3 rounded-md text-sm" /><button onClick={reanalyzeSource} disabled={loading || !sourceDraft.trim()} className="mt-3 bg-black text-white rounded-md px-4 py-2 disabled:opacity-40">Analyze corrected text</button></details>}
              <div className="flex justify-center pt-2"><button onClick={() => setActiveTab('optimize')} className="flex items-center gap-2 bg-neutral-900 hover:bg-neutral-700 text-white text-sm font-medium px-6 py-3 rounded-md transition-colors"><Sparkles className="w-4 h-4" />View Clean Resume</button></div>
            </>)}
          </div>
        )}

        {activeTab === 'graphs' && (
          <div className="space-y-6">
            {detailedResults?.score_explanation && <section className="border border-neutral-200 rounded-lg p-5 space-y-3"><h3 className="font-medium">What Changes the Coverage Score</h3><p className="text-sm text-neutral-500">{detailedResults.score_explanation.note}</p>{detailedResults.score_explanation.items.map(item => <p key={item.skill} className="text-sm">Without {item.skill}: {item.score_without_skill}/100 ({item.point_change} points lower)</p>)}</section>}
            <p className="text-sm text-neutral-500">Edges show entities mentioned on the same line. Diagrams display up to 12 nodes and 20 edges; the lists below contain all detected nodes.</p>
            {!graphData ? (<div className="flex flex-col items-center justify-center py-32 gap-4 text-center"><div className="w-20 h-20 rounded-lg bg-neutral-100 flex items-center justify-center"><Network className="w-9 h-9 text-neutral-300" /></div><p className="font-semibold text-neutral-500 text-lg">No graph data yet</p><p className="text-neutral-500 text-sm">Run an analysis to generate entity graphs</p><button onClick={() => setActiveTab('input')} className="mt-2 px-6 py-2.5 bg-neutral-900 text-white text-sm font-medium rounded-md hover:bg-neutral-700 transition-colors">Go to Analyze</button></div>) : (<>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">{[{ label: 'Resume Nodes', value: graphData.resume_graph?.nodes?.length??0, icon: <Database className="w-4 h-4" /> }, { label: 'Resume Edges', value: graphData.resume_graph?.edges?.length??0, icon: <Network className="w-4 h-4" /> }, { label: 'JD Nodes', value: graphData.jd_graph?.nodes?.length??0, icon: <Target className="w-4 h-4" /> }, { label: 'JD Edges', value: graphData.jd_graph?.edges?.length??0, icon: <Activity className="w-4 h-4" /> }].map(({ label, value, icon }, i) => (<div key={i} className="bg-white border border-neutral-200 rounded-lg p-5 text-center"><div className="flex justify-center text-neutral-500 mb-2">{icon}</div><div className="text-3xl font-bold text-neutral-900">{value}</div><div className="text-neutral-500 text-xs mt-1">{label}</div></div>))}</div>
              <div className="grid lg:grid-cols-2 gap-6">
                <div className="min-w-0 bg-white border border-neutral-200 rounded-lg p-6 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="font-semibold text-sm text-neutral-900 flex items-center gap-2"><div className="w-2.5 h-2.5 shrink-0 rounded-full bg-neutral-900" />Resume Entity Graph</h3><span className="text-xs text-neutral-500">{graphData.resume_graph?.nodes?.length??0} nodes</span></div>
                  <GraphViz nodes={graphData.resume_graph?.nodes??[]} edges={graphData.resume_graph?.edges??[]} color="#171717" />
                </div>
                <div className="min-w-0 bg-white border border-neutral-200 rounded-lg p-6 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="font-semibold text-sm text-neutral-900 flex items-center gap-2"><div className="w-2.5 h-2.5 shrink-0 rounded-full bg-neutral-900" />Job Description Entity Graph</h3><span className="text-xs text-neutral-500">{graphData.jd_graph?.nodes?.length??0} nodes</span></div>
                  <GraphViz nodes={graphData.jd_graph?.nodes??[]} edges={graphData.jd_graph?.edges??[]} color="#171717" />
                </div>
              </div>
              <div className="grid lg:grid-cols-2 gap-6">{[{ title: 'Resume Nodes', nodes: graphData.resume_graph?.nodes??[], accent: 'text-neutral-900' }, { title: 'JD Nodes', nodes: graphData.jd_graph?.nodes??[], accent: 'text-neutral-900' }].map(({ title, nodes, accent }, gi) => (<div key={gi} className="bg-white border border-neutral-200 rounded-lg p-5"><h3 className="font-semibold text-sm text-neutral-900 mb-3">{title}</h3><div className="space-y-1">{nodes.map((n,i)=>(<div key={i} className="flex items-center justify-between py-2 border-b border-neutral-50 last:border-0"><div className="flex items-center gap-2.5"><div className="w-6 h-6 rounded-lg bg-neutral-100 flex items-center justify-center">{n.type==='skill'?<Code className="w-3 h-3 text-neutral-500" />:n.type==='role'?<Briefcase className="w-3 h-3 text-neutral-500" />:<Globe className="w-3 h-3 text-neutral-500" />}</div><div><p className="text-sm font-medium text-neutral-800">{n.id}</p><p className="text-xs text-neutral-500 capitalize">{n.type}</p></div></div><span className={`text-sm font-bold ${accent}`}>Detected</span></div>))}</div></div>))}</div>
              {detailedResults?.matching && (<div className="bg-white border border-neutral-200 rounded-lg p-5"><h3 className="font-semibold text-sm text-neutral-900 mb-4">Skill Overlap Analysis</h3><div className="grid md:grid-cols-2 gap-5"><div className="space-y-2"><p className="text-xs text-neutral-500 uppercase tracking-wider font-semibold">Matching Skills</p><div className="flex flex-wrap gap-2">{detailedResults.matching.skill_overlap?.length?detailedResults.matching.skill_overlap.map((s,i)=><span key={i} className="bg-neutral-50 border border-neutral-200 text-neutral-900 text-xs px-3 py-1 rounded-md">{s}</span>):<span className="text-neutral-500 text-sm">None found</span>}</div></div><div className="space-y-2"><p className="text-xs text-neutral-500 uppercase tracking-wider font-semibold">Missing Skills</p><div className="flex flex-wrap gap-2">{detailedResults.matching.missing_skills?.length?detailedResults.matching.missing_skills.map((s,i)=><span key={i} className="bg-neutral-50 border border-neutral-200 text-neutral-900 text-xs px-3 py-1 rounded-md">{s}</span>):<span className="text-neutral-500 text-sm">None detected</span>}</div></div></div></div>)}
            </>)}
          </div>
        )}

        {activeTab === 'optimize' && (
          <div className="space-y-6">
            {!optimizedResume ? (<div className="flex flex-col items-center justify-center py-32 gap-4 text-center"><div className="w-20 h-20 rounded-lg bg-neutral-100 flex items-center justify-center"><Sparkles className="w-9 h-9 text-neutral-300" /></div><p className="font-semibold text-neutral-500 text-lg">No clean resume yet</p><p className="text-neutral-500 text-sm">Run an analysis to prepare a clean copy of your resume</p><button onClick={() => setActiveTab('input')} className="mt-2 px-6 py-2.5 bg-neutral-900 text-white text-sm font-medium rounded-md hover:bg-neutral-700 transition-colors">Go to Analyze</button></div>) : (<>
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div><h2 className="text-xl font-medium text-neutral-900">Clean Resume</h2><p className="text-neutral-500 text-sm mt-0.5">Formatting preserves content; wording changes are applied only after your review.</p></div>
                <div className="flex flex-wrap gap-3"><button onClick={() => downloadBlob(new Blob([optimizedResume], { type: 'text/plain;charset=utf-8' }), 'careerlens-resume.txt')} className="border border-neutral-200 text-sm px-4 py-3 rounded-md">Download text</button><button onClick={() => window.print()} className="border border-neutral-200 text-sm px-4 py-3 rounded-md">Print</button><button onClick={downloadAsPDF} className="flex items-center gap-2 bg-neutral-900 hover:bg-neutral-700 text-white text-sm font-semibold px-5 py-3 rounded-md transition-colors">
                  <Download className="w-4 h-4" />Download as PDF
                </button></div>
              </div>
              <section className="border border-neutral-200 rounded-lg p-5 space-y-4"><h3 className="font-medium">Review Proposed Changes</h3><p className="text-sm text-neutral-500">Accept only the changes you want. Unchecked edits keep their original wording. Section templates reorder existing content without adding qualifications.</p><label className="text-sm flex items-center gap-3">Section order<select value={template} onChange={e => setTemplate(e.target.value)} className="border border-neutral-200 p-2 rounded-md">{optimization?.templates?.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>{optimization?.proposals?.length ? optimization.proposals.map(edit => <div key={edit.id} className="border-t border-neutral-100 pt-4 space-y-2"><label className="flex items-start gap-3 text-sm"><input type="checkbox" checked={acceptedEdits.includes(edit.id)} onChange={e => setAcceptedEdits(e.target.checked ? [...acceptedEdits, edit.id] : acceptedEdits.filter(id => id !== edit.id))} /><span>{edit.reason} · line {edit.line}</span></label><div className="grid sm:grid-cols-2 gap-3 text-sm"><div className="p-3 bg-neutral-50"><p className="text-xs uppercase tracking-wide text-neutral-500 mb-2">Original / source evidence</p>{edit.original}</div><div className="p-3 border border-neutral-200"><p className="text-xs uppercase tracking-wide text-neutral-500 mb-2">Proposed</p>{edit.proposed}</div></div></div>) : <p className="text-sm text-neutral-500">No supported wording changes found. Formatting and section ordering are still available.</p>}<button disabled={editLoading || proposalLoading} onClick={previewEdits} className="bg-black text-white text-sm px-5 py-3 rounded-md disabled:opacity-40">{editLoading ? 'Applying...' : 'Apply selected changes & reanalyze'}</button></section>
              <section className="border border-neutral-200 rounded-lg p-5 space-y-3"><h3 className="font-medium">Local Wording Suggestions</h3><p className="text-sm text-neutral-500">Generate alternatives locally, then review both versions. Exact approved transformations preserve the factual suffix; free-form edits also require model comparison.</p><label className="text-xs text-neutral-600">Proposal model<select aria-label="Proposal model" value={rewriteEngine} onChange={e => setRewriteEngine(e.target.value)} className="block border border-neutral-200 p-2 rounded mt-1"><option value="gemma">Local Gemma 4 E4B</option><option value="t5">Experimental T5 + Gemma validation</option></select></label><button onClick={generateLocalEdits} disabled={proposalLoading || editLoading || !systemStatus?.local_ai?.available} className="bg-black text-white rounded-md px-4 py-2 text-sm disabled:opacity-40">{proposalLoading ? 'Generating and checking facts…' : 'Generate local suggestions'}</button>{proposalNote && <p role="status" className="text-sm text-neutral-600">{proposalNote}</p>}</section>
              {comparison && <p className="text-sm text-neutral-600">Measured skill coverage: {comparison.before ?? 'N/A'} → {comparison.after ?? 'N/A'}; change: {comparison.delta ?? 'N/A'} points. Wording improvements can have zero score change.</p>}
              <p className="text-sm text-neutral-500">{optimization?.note}</p>
              <div className="bg-white border border-neutral-200 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2 text-xs text-neutral-500 font-mono"><div className="w-2 h-2 rounded-full bg-neutral-900" />careerlens-resume.pdf</div>
                  <span className="text-xs text-neutral-500 bg-neutral-100 px-2 py-1 rounded-full">Preview</span>
                </div>
                <pre className="whitespace-pre-wrap text-sm text-neutral-700 font-mono leading-relaxed bg-neutral-50 rounded-md p-5 border border-neutral-100">{optimizedResume}</pre>
              </div>
              <div className="bg-white border border-neutral-200 rounded-lg p-5"><h3 className="font-semibold text-sm mb-3">Applied Formatting Changes</h3>{optimization?.changes.length ? optimization.changes.map((change, i) => <p key={i} className="text-sm text-neutral-600">{change}</p>) : <p className="text-sm text-neutral-500">Your input already uses clean spacing and bullet markers.</p>}</div>
            </>)}
          </div>
        )}

      </main>
      <footer className="border-t border-neutral-200 mt-20 bg-white">
        <div className="max-w-6xl mx-auto px-5 sm:px-8 py-6 flex flex-col md:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-neutral-500 text-sm"><div className="w-5 h-5 rounded-lg bg-neutral-900 flex items-center justify-center"><Brain className="w-3 h-3 text-white" /></div>CareerLens - Resume Analysis</div>
          <div className="flex flex-wrap justify-center items-center gap-3 text-xs text-neutral-500"><span>Entity Graphs</span><span>Skill Coverage</span><span>Job Wording Review</span></div>
          <Link href="/review" className="text-xs underline text-neutral-600">Review training examples</Link>
        </div>
      </footer>
    </div>
    </>
  )
}
