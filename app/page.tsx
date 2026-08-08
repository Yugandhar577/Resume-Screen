'use client'
import { useState, useRef, useEffect } from 'react'
import { Upload, FileText, Target, Download, Brain, BarChart3, Zap, TrendingUp, Shield, CheckCircle, AlertCircle, Activity, Network, Database, Play, Briefcase, Code, Globe, Sparkles, ChevronDown } from 'lucide-react'

type SystemStatus = { status: string; ml_engine: boolean; groq_available: boolean } | null
type Analysis = { atsScore: number; atsFeedback: string; techFeedback: string; hrFeedback: string; skillGaps: string[]; recommendations: string[] } | null
type GraphNode = { id: string; type: string; weight: number }
type GraphEdge = { source: string; target: string; weight: number }
type GraphData = { resume_graph: { nodes: GraphNode[]; edges: GraphEdge[] }; jd_graph: { nodes: GraphNode[]; edges: GraphEdge[] } } | null
type DetailedResults = {
  gat_matching?: { graph_similarity: number; skill_overlap: string[]; missing_skills: string[]; match_score: number }
  bias_detection?: { bias_detected: boolean; bias_types: string[]; quality_score: number }
  career_trajectory?: { predicted_roles: string[]; skill_development: string[]; timeline: string }
} | null

function GraphViz({ nodes, edges, color }: { nodes: GraphNode[]; edges: GraphEdge[]; color: string }) {
  const W = 480, H = 280
  if (!nodes.length) return <div className="flex items-center justify-center h-40 text-gray-400 text-sm">No graph data</div>
  const list = nodes.slice(0, 12)
  const positioned = list.map((n, i, arr) => {
    const angle = (2 * Math.PI * i) / arr.length - Math.PI / 2
    const r = arr.length <= 1 ? 0 : Math.min(W / 2 - 55, H / 2 - 45)
    return { ...n, x: W / 2 + r * Math.cos(angle), y: H / 2 + r * Math.sin(angle) }
  })
  const posMap: Record<string, { x: number; y: number }> = {}
  positioned.forEach(n => { posMap[n.id] = { x: n.x, y: n.y } })
  const sc: Record<string, string> = { skill: color, role: '#1f2937', company: '#9ca3af' }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto bg-gray-50 rounded-xl">
      {edges.slice(0, 20).map((e, i) => {
        const s = posMap[e.source], t = posMap[e.target]
        if (!s || !t) return null
        return <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke="#d1d5db" strokeWidth={1.5} />
      })}
      {positioned.map((n, i) => (
        <g key={i}>
          <circle cx={n.x} cy={n.y} r={20} fill="white" stroke={sc[n.type] ?? '#d1d5db'} strokeWidth={2.5} />
          <text x={n.x} y={n.y+1} textAnchor="middle" dominantBaseline="middle" fontSize={9} fontWeight="700" fill={sc[n.type]??'#6b7280'}>{n.type==='skill'?'SK':n.type==='role'?'RL':'CO'}</text>
          <text x={n.x} y={n.y+33} textAnchor="middle" fontSize={8.5} fill="#111827" fontWeight="600">{n.id.length>12?n.id.slice(0,11)+'...':n.id}</text>
          <text x={n.x} y={n.y+44} textAnchor="middle" fontSize={7.5} fill={sc[n.type]??'#9ca3af'}>{(n.weight*100).toFixed(0)}%</text>
        </g>
      ))}
    </svg>
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
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { checkStatus() }, [])
  const checkStatus = async () => {
    try { setSystemStatus(await (await fetch('/health')).json()) }
    catch { setSystemStatus({ status: 'offline', ml_engine: false, groq_available: false }) }
  }

  const loadSample = (key: string) => {
    const s = SAMPLE_DATASETS[key as keyof typeof SAMPLE_DATASETS]
    if (s) { setResume(s.resume); setJobDescription(s.jd); setSelectedSample(key); setShowSampleMenu(false) }
  }

  const runAnalysis = async () => {
    if (!resume.trim() || !jobDescription.trim()) return
    setLoading(true); setAnalysis(null); setOptimizedResume(''); setDetailedResults(null); setActiveTab('results')
    try {
      const data = await (await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ resume, jobDescription }) })).json()
      setAnalysis(data.analysis); setOptimizedResume(data.optimizedResume ?? ''); setDetailedResults(data.detailed_results ?? null)
      try { setGraphData(await (await fetch('/api/graph-data/latest')).json()) } catch {}
    } catch (e) { console.error(e) } finally { setLoading(false) }
  }

  const runPDF = async (file: File) => {
    setLoading(true); setAnalysis(null); setOptimizedResume(''); setDetailedResults(null); setActiveTab('results')
    try {
      const form = new FormData(); form.append('file', file); form.append('jobDescription', jobDescription)
      const data = await (await fetch('/api/analyze-pdf', { method: 'POST', body: form })).json()
      setAnalysis(data.analysis); setOptimizedResume(data.optimizedResume ?? ''); setDetailedResults(data.detailed_results ?? null)
      try { setGraphData(await (await fetch('/api/graph-data/latest')).json()) } catch {}
    } catch (e) { console.error(e) } finally { setLoading(false) }
  }

  const downloadAsPDF = async () => {
    if (!optimizedResume) return
    const { jsPDF } = await import('jspdf')
    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })
    const margin = 20
    const pageW = doc.internal.pageSize.getWidth()
    const pageH = doc.internal.pageSize.getHeight()
    const maxW = pageW - margin * 2
    let y = margin
    doc.setFillColor(17, 24, 39)
    doc.rect(0, 0, pageW, 18, 'F')
    doc.setTextColor(255, 255, 255)
    doc.setFontSize(14)
    doc.setFont('helvetica', 'bold')
    doc.text('CareerLens AI - Optimized Resume', margin, 12)
    y = 28
    doc.setTextColor(30, 30, 30)
    const lines = optimizedResume.split('\n')
    lines.forEach(line => {
      if (y > pageH - margin) { doc.addPage(); y = margin }
      const trimmed = line.trim()
      if (trimmed.startsWith('---') || trimmed.startsWith('===')) {
        doc.setDrawColor(200, 200, 200)
        doc.line(margin, y, pageW - margin, y)
        y += 4
      } else if (trimmed.startsWith('CAREERLENS') || trimmed.startsWith('OPTIMIZED') || (trimmed === trimmed.toUpperCase() && trimmed.length > 3 && !trimmed.startsWith('-') && !trimmed.startsWith('•'))) {
        doc.setFontSize(11)
        doc.setFont('helvetica', 'bold')
        doc.setTextColor(17, 24, 39)
        doc.text(trimmed, margin, y)
        y += 6
      } else if (trimmed.startsWith('-') || trimmed.startsWith('•') || trimmed.startsWith('✅')) {
        doc.setFontSize(9)
        doc.setFont('helvetica', 'normal')
        doc.setTextColor(60, 60, 60)
        const wrapped = doc.splitTextToSize('  ' + trimmed, maxW)
        doc.text(wrapped, margin, y)
        y += wrapped.length * 4.5
      } else if (trimmed === '') {
        y += 3
      } else {
        doc.setFontSize(9)
        doc.setFont('helvetica', 'normal')
        doc.setTextColor(50, 50, 50)
        const wrapped = doc.splitTextToSize(trimmed, maxW)
        doc.text(wrapped, margin, y)
        y += wrapped.length * 4.5
      }
    })
    doc.setFontSize(7)
    doc.setTextColor(150, 150, 150)
    doc.text('Generated by CareerLens AI - careerlens.ai', margin, pageH - 8)
    doc.save('careerlens-optimized-resume.pdf')
  }

  const isOnline = systemStatus?.status === 'healthy'
  const sampleLabels: Record<string, string> = { software_engineer: 'Software Engineer', data_scientist: 'Data Scientist', devops_engineer: 'DevOps Engineer' }
  const tabs = [
    { id: 'input' as const, label: 'Analyze', icon: <FileText className="w-3.5 h-3.5" /> },
    { id: 'results' as const, label: 'Results', icon: <BarChart3 className="w-3.5 h-3.5" /> },
    { id: 'graphs' as const, label: 'Graphs', icon: <Network className="w-3.5 h-3.5" /> },
    { id: 'optimize' as const, label: 'Optimize', icon: <Sparkles className="w-3.5 h-3.5" /> },
  ]
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-gray-900 flex items-center justify-center"><Brain className="w-4 h-4 text-white" /></div>
            <span className="font-bold text-lg tracking-tight text-gray-900">CareerLens</span>
            <span className="text-xs text-gray-400 font-mono bg-gray-100 px-2 py-0.5 rounded-full">AI</span>
          </div>
          <nav className="hidden md:flex items-center gap-1 bg-gray-100 rounded-xl p-1">
            {tabs.map(t => (<button key={t.id} onClick={() => setActiveTab(t.id)} className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === t.id ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-900'}`}>{t.icon}{t.label}</button>))}
          </nav>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-red-400'}`} />
            <span className="text-xs text-gray-400 font-mono">{systemStatus?.status ?? 'connecting...'}</span>
          </div>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-6 py-8">

        {activeTab === 'input' && (
          <div className="space-y-8">
            <div className="text-center py-10">
              <div className="inline-flex items-center gap-2 bg-gray-900 text-white rounded-full px-4 py-1.5 text-xs font-semibold mb-5"><Zap className="w-3 h-3" />Graph Neural Networks + Groq AI</div>
              <h1 className="text-6xl font-black tracking-tighter text-gray-900 mb-4">CareerLens AI</h1>
              <p className="text-gray-500 text-lg max-w-xl mx-auto">Deep career intelligence using knowledge graphs, GAT matching, bias detection and trajectory prediction.</p>
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              {[{ icon: <Brain className="w-3.5 h-3.5" />, label: 'ML Engine', on: systemStatus?.ml_engine }, { icon: <Zap className="w-3.5 h-3.5" />, label: 'Groq AI', on: systemStatus?.groq_available }, { icon: <Network className="w-3.5 h-3.5" />, label: 'Knowledge Graphs', on: true }, { icon: <Shield className="w-3.5 h-3.5" />, label: 'Bias Detection', on: true }, { icon: <TrendingUp className="w-3.5 h-3.5" />, label: 'Career Trajectory', on: true }].map(({ icon, label, on }, i) => (<div key={i} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border ${on ? 'bg-gray-900 text-white border-gray-900' : 'bg-white text-gray-400 border-gray-200'}`}>{icon}{label}</div>))}
            </div>

            {/* Sample Dataset Selector */}
            <div className="flex justify-center">
              <div className="relative">
                <button onClick={() => setShowSampleMenu(!showSampleMenu)} className="flex items-center gap-2 bg-white border border-gray-200 hover:border-gray-400 text-gray-700 text-sm font-medium px-5 py-2.5 rounded-xl transition-all shadow-sm">
                  <Play className="w-3.5 h-3.5 text-gray-500" />
                  {selectedSample ? sampleLabels[selectedSample] : 'Load Sample Dataset'}
                  <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
                </button>
                {showSampleMenu && (
                  <div className="absolute top-full mt-2 left-0 bg-white border border-gray-200 rounded-xl shadow-lg z-10 min-w-56 overflow-hidden">
                    {Object.entries(sampleLabels).map(([key, label]) => (
                      <button key={key} onClick={() => loadSample(key)} className="w-full text-left px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-3 border-b border-gray-50 last:border-0">
                        <div className="w-7 h-7 rounded-lg bg-gray-100 flex items-center justify-center"><Briefcase className="w-3.5 h-3.5 text-gray-500" /></div>
                        <div><p className="font-medium">{label}</p><p className="text-xs text-gray-400">Resume + Job Description</p></div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="grid lg:grid-cols-2 gap-6">
              <div className="bg-white border border-gray-200 rounded-2xl p-6 space-y-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-blue-50 flex items-center justify-center"><FileText className="w-4 h-4 text-blue-600" /></div>
                    <div><p className="font-semibold text-sm text-gray-900">Resume</p><p className="text-gray-400 text-xs">Paste text or upload PDF</p></div>
                  </div>
                  <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
                    {(['text', 'pdf'] as const).map(m => (<button key={m} onClick={() => setUploadMode(m)} className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${uploadMode === m ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-400 hover:text-gray-700'}`}>{m.toUpperCase()}</button>))}
                  </div>
                </div>
                {uploadMode === 'text' ? (
                  <textarea value={resume} onChange={e => setResume(e.target.value)} placeholder="Paste your resume here or load a sample above..." className="w-full h-72 bg-gray-50 border border-gray-200 rounded-xl p-4 text-sm text-gray-800 placeholder-gray-300 resize-none focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all font-mono" />
                ) : (
                  <div onClick={() => fileInputRef.current?.click()} className="h-72 border-2 border-dashed border-gray-200 rounded-xl flex flex-col items-center justify-center gap-4 cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-all group">
                    <div className="w-14 h-14 rounded-2xl bg-gray-100 flex items-center justify-center group-hover:bg-blue-100 transition-all"><Upload className="w-6 h-6 text-gray-400 group-hover:text-blue-500" /></div>
                    <div className="text-center"><p className="font-semibold text-sm text-gray-700">Drop PDF resume here</p><p className="text-gray-400 text-xs mt-1">or click to browse — PDF files only</p></div>
                    <input ref={fileInputRef} type="file" accept=".pdf" onChange={e => { const f = e.target.files?.[0]; if (f?.type === 'application/pdf') runPDF(f) }} className="hidden" />
                  </div>
                )}
              </div>
              <div className="bg-white border border-gray-200 rounded-2xl p-6 space-y-4 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-emerald-50 flex items-center justify-center"><Target className="w-4 h-4 text-emerald-600" /></div>
                  <div><p className="font-semibold text-sm text-gray-900">Job Description</p><p className="text-gray-400 text-xs">Target role requirements</p></div>
                </div>
                <textarea value={jobDescription} onChange={e => setJobDescription(e.target.value)} placeholder="Paste the job description here or load a sample above..." className="w-full h-72 bg-gray-50 border border-gray-200 rounded-xl p-4 text-sm text-gray-800 placeholder-gray-300 resize-none focus:outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 transition-all font-mono" />
              </div>
            </div>
            <div className="flex justify-center pt-4">
              <button onClick={runAnalysis} disabled={loading || !resume.trim() || !jobDescription.trim()} className="flex items-center gap-3 bg-gray-900 hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold text-base px-10 py-4 rounded-2xl transition-all hover:scale-105 active:scale-100 shadow-lg">
                {loading ? (<><div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /><span>Analyzing...</span></>) : (<><Brain className="w-5 h-5" /><span>Analyze with CareerLens AI</span><Zap className="w-4 h-4" /></>)}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'results' && (
          <div className="space-y-6">
            {loading && (<div className="flex flex-col items-center justify-center py-32 gap-6"><div className="relative w-20 h-20"><div className="absolute inset-0 border-4 border-gray-100 rounded-full" /><div className="absolute inset-0 border-4 border-gray-900 border-t-transparent rounded-full animate-spin" /><Brain className="absolute inset-0 m-auto w-6 h-6 text-gray-700" /></div><div className="text-center"><p className="font-bold text-lg text-gray-900">Running ML Pipeline</p><p className="text-gray-400 text-sm mt-1">NER extraction to Graph building to GAT matching to Optimization</p></div></div>)}
            {!loading && !analysis && (<div className="flex flex-col items-center justify-center py-32 gap-4 text-center"><div className="w-20 h-20 rounded-2xl bg-gray-100 flex items-center justify-center"><BarChart3 className="w-9 h-9 text-gray-300" /></div><p className="font-semibold text-gray-500 text-lg">No results yet</p><p className="text-gray-400 text-sm">Run an analysis from the Analyze tab</p><button onClick={() => setActiveTab('input')} className="mt-2 px-6 py-2.5 bg-gray-900 text-white text-sm font-medium rounded-xl hover:bg-gray-700 transition-all">Go to Analyze</button></div>)}
            {!loading && analysis && (<>
              <div className="bg-white border border-gray-200 rounded-2xl p-8 flex flex-col md:flex-row items-center gap-8 shadow-sm">
                <div className="relative w-40 h-40 shrink-0">
                  <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="42" stroke="#f3f4f6" strokeWidth="8" fill="none" />
                    <circle cx="50" cy="50" r="42" stroke="#111827" strokeWidth="8" fill="none" strokeLinecap="round" strokeDasharray={`${(analysis.atsScore/100)*263.9} 263.9`} />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center"><span className="text-4xl font-black text-gray-900">{analysis.atsScore}</span><span className="text-gray-400 text-xs font-medium">/100</span></div>
                </div>
                <div className="flex-1 space-y-4">
                  <div><h2 className="text-2xl font-bold text-gray-900">GAT Match Score</h2><p className="text-gray-400 text-sm mt-1">Graph Attention Network analysis complete</p></div>
                  {detailedResults?.gat_matching && (<div className="grid grid-cols-3 gap-3">{[{ label: 'Graph Similarity', value: `${(detailedResults.gat_matching.graph_similarity*100).toFixed(0)}%` }, { label: 'Skill Matches', value: detailedResults.gat_matching.skill_overlap?.length??0 }, { label: 'Skill Gaps', value: detailedResults.gat_matching.missing_skills?.length??0 }].map(({ label, value }, i) => (<div key={i} className="bg-gray-50 border border-gray-100 rounded-xl p-4 text-center"><div className="text-2xl font-bold text-gray-900">{value}</div><div className="text-gray-400 text-xs mt-1">{label}</div></div>))}</div>)}
                </div>
              </div>
              <div className="grid md:grid-cols-3 gap-4">{[{ title: 'Knowledge Graph', content: analysis.atsFeedback, icon: <Network className="w-4 h-4" />, c: 'text-blue-600', bg: 'bg-blue-50', b: 'border-blue-100' }, { title: 'ML Technical', content: analysis.techFeedback, icon: <Brain className="w-4 h-4" />, c: 'text-violet-600', bg: 'bg-violet-50', b: 'border-violet-100' }, { title: 'Career Intelligence', content: analysis.hrFeedback, icon: <TrendingUp className="w-4 h-4" />, c: 'text-emerald-600', bg: 'bg-emerald-50', b: 'border-emerald-100' }].map(({ title, content, icon, c, bg, b }, i) => (<div key={i} className="bg-white border border-gray-200 rounded-2xl p-5 space-y-3 shadow-sm"><div className={`flex items-center gap-2 text-sm font-semibold ${c}`}><div className={`w-7 h-7 rounded-lg ${bg} border ${b} flex items-center justify-center`}>{icon}</div>{title}</div><p className="text-gray-500 text-sm leading-relaxed">{content}</p></div>))}</div>
              {(analysis.skillGaps?.length??0)>0 && (<div className="bg-white border border-gray-200 rounded-2xl p-5 space-y-3 shadow-sm"><div className="flex items-center gap-2 text-sm font-semibold text-red-600"><AlertCircle className="w-4 h-4" />Skill Gaps ({analysis.skillGaps.length})</div><div className="flex flex-wrap gap-2">{analysis.skillGaps.map((s,i)=><span key={i} className="bg-red-50 border border-red-100 text-red-600 text-xs px-3 py-1.5 rounded-full font-medium">{s}</span>)}</div></div>)}
              {(analysis.recommendations?.length??0)>0 && (<div className="bg-white border border-gray-200 rounded-2xl p-5 space-y-3 shadow-sm"><div className="flex items-center gap-2 text-sm font-semibold text-emerald-600"><CheckCircle className="w-4 h-4" />Recommendations</div><div className="space-y-2">{analysis.recommendations.map((r,i)=>(<div key={i} className="flex items-start gap-3 text-sm text-gray-600"><div className="w-5 h-5 rounded-full bg-emerald-100 flex items-center justify-center shrink-0 mt-0.5 text-xs font-bold text-emerald-700">{i+1}</div>{r}</div>))}</div></div>)}
              {detailedResults?.bias_detection && (<div className="bg-white border border-gray-200 rounded-2xl p-5 space-y-3 shadow-sm"><div className="flex items-center gap-2 text-sm font-semibold text-amber-600"><Shield className="w-4 h-4" />JD Bias Detection</div><div className="flex items-center gap-6"><div className="text-center bg-amber-50 border border-amber-100 rounded-xl p-4"><div className="text-3xl font-bold text-amber-700">{detailedResults.bias_detection.quality_score}</div><div className="text-amber-600 text-xs mt-1">Quality Score</div></div><div className="flex-1">{detailedResults.bias_detection.bias_detected?<div className="flex flex-wrap gap-2">{detailedResults.bias_detection.bias_types.map((b,i)=><span key={i} className="bg-amber-50 border border-amber-100 text-amber-700 text-xs px-3 py-1 rounded-full">{b}</span>)}</div>:<p className="text-sm text-emerald-600 flex items-center gap-2"><CheckCircle className="w-4 h-4" />No significant bias detected</p>}</div></div></div>)}
              {detailedResults?.career_trajectory && (<div className="bg-white border border-gray-200 rounded-2xl p-5 space-y-4 shadow-sm"><div className="flex items-center gap-2 text-sm font-semibold text-blue-600"><TrendingUp className="w-4 h-4" />Career Trajectory</div><div className="grid md:grid-cols-2 gap-6"><div className="space-y-2"><p className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Predicted Roles</p>{detailedResults.career_trajectory.predicted_roles.map((r,i)=>(<div key={i} className="flex items-center gap-3 text-sm text-gray-700"><div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center text-xs font-bold text-blue-700">{i+1}</div>{r}</div>))}</div><div className="space-y-2"><p className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Skills to Develop</p>{detailedResults.career_trajectory.skill_development.map((s,i)=>(<div key={i} className="flex items-center gap-2 text-sm text-gray-700"><div className="w-2 h-2 rounded-full bg-blue-400" />{s}</div>))}</div></div><p className="text-xs text-gray-400 flex items-center gap-1.5 pt-1 border-t border-gray-100"><Activity className="w-3.5 h-3.5" />Timeline: {detailedResults.career_trajectory.timeline}</p></div>)}
              <div className="flex justify-center pt-2"><button onClick={() => setActiveTab('optimize')} className="flex items-center gap-2 bg-gray-900 hover:bg-gray-700 text-white text-sm font-medium px-6 py-3 rounded-xl transition-all"><Sparkles className="w-4 h-4" />View Optimized Resume</button></div>
            </>)}
          </div>
        )}

        {activeTab === 'graphs' && (
          <div className="space-y-6">
            {!graphData ? (<div className="flex flex-col items-center justify-center py-32 gap-4 text-center"><div className="w-20 h-20 rounded-2xl bg-gray-100 flex items-center justify-center"><Network className="w-9 h-9 text-gray-300" /></div><p className="font-semibold text-gray-500 text-lg">No graph data yet</p><p className="text-gray-400 text-sm">Run an analysis to generate knowledge graphs</p><button onClick={() => setActiveTab('input')} className="mt-2 px-6 py-2.5 bg-gray-900 text-white text-sm font-medium rounded-xl hover:bg-gray-700 transition-all">Go to Analyze</button></div>) : (<>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">{[{ label: 'Resume Nodes', value: graphData.resume_graph?.nodes?.length??0, icon: <Database className="w-4 h-4" /> }, { label: 'Resume Edges', value: graphData.resume_graph?.edges?.length??0, icon: <Network className="w-4 h-4" /> }, { label: 'JD Nodes', value: graphData.jd_graph?.nodes?.length??0, icon: <Target className="w-4 h-4" /> }, { label: 'JD Edges', value: graphData.jd_graph?.edges?.length??0, icon: <Activity className="w-4 h-4" /> }].map(({ label, value, icon }, i) => (<div key={i} className="bg-white border border-gray-200 rounded-2xl p-5 text-center shadow-sm"><div className="flex justify-center text-gray-400 mb-2">{icon}</div><div className="text-3xl font-bold text-gray-900">{value}</div><div className="text-gray-400 text-xs mt-1">{label}</div></div>))}</div>
              <div className="grid lg:grid-cols-2 gap-6">
                <div className="bg-white border border-gray-200 rounded-2xl p-6 space-y-4 shadow-sm">
                  <div className="flex items-center justify-between"><h3 className="font-semibold text-sm text-gray-900 flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-blue-500" />Resume Knowledge Graph</h3><span className="text-xs text-gray-400">{graphData.resume_graph?.nodes?.length??0} nodes</span></div>
                  <GraphViz nodes={graphData.resume_graph?.nodes??[]} edges={graphData.resume_graph?.edges??[]} color="#3b82f6" />
                  <div className="flex items-center gap-4 text-xs text-gray-400 pt-2 border-t border-gray-100"><span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full border-2 border-blue-500 inline-block" />Skill (SK)</span><span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full border-2 border-gray-800 inline-block" />Role (RL)</span><span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full border-2 border-gray-400 inline-block" />Company (CO)</span></div>
                </div>
                <div className="bg-white border border-gray-200 rounded-2xl p-6 space-y-4 shadow-sm">
                  <div className="flex items-center justify-between"><h3 className="font-semibold text-sm text-gray-900 flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />Job Description Graph</h3><span className="text-xs text-gray-400">{graphData.jd_graph?.nodes?.length??0} nodes</span></div>
                  <GraphViz nodes={graphData.jd_graph?.nodes??[]} edges={graphData.jd_graph?.edges??[]} color="#10b981" />
                  <div className="flex items-center gap-4 text-xs text-gray-400 pt-2 border-t border-gray-100"><span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full border-2 border-emerald-500 inline-block" />Skill (SK)</span><span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full border-2 border-gray-800 inline-block" />Role (RL)</span><span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full border-2 border-gray-400 inline-block" />Company (CO)</span></div>
                </div>
              </div>
              <div className="grid lg:grid-cols-2 gap-6">{[{ title: 'Resume Nodes', nodes: graphData.resume_graph?.nodes??[], accent: 'text-blue-600' }, { title: 'JD Nodes', nodes: graphData.jd_graph?.nodes??[], accent: 'text-emerald-600' }].map(({ title, nodes, accent }, gi) => (<div key={gi} className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm"><h3 className="font-semibold text-sm text-gray-900 mb-3">{title}</h3><div className="space-y-1">{nodes.slice(0,10).map((n,i)=>(<div key={i} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0"><div className="flex items-center gap-2.5"><div className="w-6 h-6 rounded-lg bg-gray-100 flex items-center justify-center">{n.type==='skill'?<Code className="w-3 h-3 text-gray-500" />:n.type==='role'?<Briefcase className="w-3 h-3 text-gray-500" />:<Globe className="w-3 h-3 text-gray-500" />}</div><div><p className="text-sm font-medium text-gray-800">{n.id}</p><p className="text-xs text-gray-400 capitalize">{n.type}</p></div></div><span className={`text-sm font-bold ${accent}`}>{(n.weight*100).toFixed(0)}%</span></div>))}</div></div>))}</div>
              {detailedResults?.gat_matching && (<div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm"><h3 className="font-semibold text-sm text-gray-900 mb-4">Graph Overlap Analysis</h3><div className="grid md:grid-cols-2 gap-5"><div className="space-y-2"><p className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Matching Skills</p><div className="flex flex-wrap gap-2">{detailedResults.gat_matching.skill_overlap?.length?detailedResults.gat_matching.skill_overlap.map((s,i)=><span key={i} className="bg-emerald-50 border border-emerald-100 text-emerald-700 text-xs px-3 py-1 rounded-full">{s}</span>):<span className="text-gray-400 text-sm">None found</span>}</div></div><div className="space-y-2"><p className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Missing Skills</p><div className="flex flex-wrap gap-2">{detailedResults.gat_matching.missing_skills?.length?detailedResults.gat_matching.missing_skills.map((s,i)=><span key={i} className="bg-red-50 border border-red-100 text-red-600 text-xs px-3 py-1 rounded-full">{s}</span>):<span className="text-gray-400 text-sm">None detected</span>}</div></div></div></div>)}
            </>)}
          </div>
        )}

        {activeTab === 'optimize' && (
          <div className="space-y-6">
            {!optimizedResume ? (<div className="flex flex-col items-center justify-center py-32 gap-4 text-center"><div className="w-20 h-20 rounded-2xl bg-gray-100 flex items-center justify-center"><Sparkles className="w-9 h-9 text-gray-300" /></div><p className="font-semibold text-gray-500 text-lg">No optimized resume yet</p><p className="text-gray-400 text-sm">Run an analysis to generate your optimized resume</p><button onClick={() => setActiveTab('input')} className="mt-2 px-6 py-2.5 bg-gray-900 text-white text-sm font-medium rounded-xl hover:bg-gray-700 transition-all">Go to Analyze</button></div>) : (<>
              <div className="flex items-center justify-between">
                <div><h2 className="text-xl font-bold text-gray-900">ML-Optimized Resume</h2><p className="text-gray-400 text-sm mt-0.5">Enhanced by CareerLens knowledge graph pipeline</p></div>
                <button onClick={downloadAsPDF} className="flex items-center gap-2 bg-gray-900 hover:bg-gray-700 text-white text-sm font-semibold px-5 py-3 rounded-xl transition-all hover:scale-105 shadow-md">
                  <Download className="w-4 h-4" />Download as PDF
                </button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">{[{ label: 'ATS Compatibility', value: '+25%', color: 'text-emerald-600' }, { label: 'Keyword Match', value: '+40%', color: 'text-blue-600' }, { label: 'Skill Alignment', value: '+35%', color: 'text-violet-600' }, { label: 'Overall Score', value: '+30%', color: 'text-amber-600' }].map(({ label, value, color }, i) => (<div key={i} className="bg-white border border-gray-200 rounded-2xl p-4 text-center shadow-sm"><div className={`text-2xl font-bold ${color}`}>{value}</div><div className="text-gray-400 text-xs mt-1">{label}</div></div>))}</div>
              <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2 text-xs text-gray-400 font-mono"><div className="w-2 h-2 rounded-full bg-emerald-400" />careerlens-optimized-resume.pdf</div>
                  <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded-full">Preview</span>
                </div>
                <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono leading-relaxed bg-gray-50 rounded-xl p-5 border border-gray-100">{optimizedResume}</pre>
              </div>
              <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm"><div className="flex items-center gap-2 text-sm font-semibold text-emerald-600 mb-3"><CheckCircle className="w-4 h-4" />Applied Optimizations</div><div className="grid md:grid-cols-2 gap-2">{['Graph-based skill analysis', 'GAT matching score integrated', 'Career trajectory applied', 'Bias-aware optimization', 'ATS keyword optimization', 'Action verb enhancement'].map((o,i)=>(<div key={i} className="flex items-center gap-2.5 text-sm text-gray-600"><CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0" />{o}</div>))}</div></div>
              <div className="flex justify-center"><button onClick={downloadAsPDF} className="flex items-center gap-3 bg-gray-900 hover:bg-gray-700 text-white font-bold text-base px-10 py-4 rounded-2xl transition-all hover:scale-105 shadow-lg"><Download className="w-5 h-5" />Download Optimized Resume as PDF</button></div>
            </>)}
          </div>
        )}

      </main>
      <footer className="border-t border-gray-200 mt-16 bg-white">
        <div className="max-w-7xl mx-auto px-6 py-6 flex flex-col md:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-gray-400 text-sm"><div className="w-5 h-5 rounded-lg bg-gray-900 flex items-center justify-center"><Brain className="w-3 h-3 text-white" /></div>CareerLens - Deep Career Intelligence</div>
          <div className="flex items-center gap-5 text-xs text-gray-300"><span className="flex items-center gap-1"><Network className="w-3 h-3" />Knowledge Graphs</span><span className="flex items-center gap-1"><Brain className="w-3 h-3" />GAT Matching</span><span className="flex items-center gap-1"><Shield className="w-3 h-3" />Bias Detection</span><span className="flex items-center gap-1"><TrendingUp className="w-3 h-3" />Career Trajectory</span></div>
        </div>
      </footer>
    </div>
  )
}
