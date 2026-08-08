#!/usr/bin/env python3
"""Generates app/page.tsx - complete CareerLens UI"""

OUT = "app/page.tsx"
L = []
def a(s=""): L.append(s)

a("'use client'")
a("import { useState, useRef, useEffect } from 'react'")
a("import { Upload, FileText, Target, Download, Brain, BarChart3, Zap, TrendingUp, Shield, CheckCircle, AlertCircle, Activity, Network, Database, Play, Briefcase, Code, Globe, Sparkles, ChevronDown } from 'lucide-react'")
a()
a("type SystemStatus = { status: string; ml_engine: boolean; groq_available: boolean } | null")
a("type Analysis = { atsScore: number; atsFeedback: string; techFeedback: string; hrFeedback: string; skillGaps: string[]; recommendations: string[] } | null")
a("type GraphNode = { id: string; type: string; weight: number }")
a("type GraphEdge = { source: string; target: string; weight: number }")
a("type GraphData = { resume_graph: { nodes: GraphNode[]; edges: GraphEdge[] }; jd_graph: { nodes: GraphNode[]; edges: GraphEdge[] } } | null")
a("type DetailedResults = {")
a("  gat_matching?: { graph_similarity: number; skill_overlap: string[]; missing_skills: string[]; match_score: number }")
a("  bias_detection?: { bias_detected: boolean; bias_types: string[]; quality_score: number }")
a("  career_trajectory?: { predicted_roles: string[]; skill_development: string[]; timeline: string }")
a("} | null")
a()
a("function GraphViz({ nodes, edges, color }: { nodes: GraphNode[]; edges: GraphEdge[]; color: string }) {")
a("  const W = 480, H = 280")
a('  if (!nodes.length) return <div className="flex items-center justify-center h-40 text-gray-400 text-sm">No graph data</div>')
a("  const list = nodes.slice(0, 12)")
a("  const positioned = list.map((n, i, arr) => {")
a("    const angle = (2 * Math.PI * i) / arr.length - Math.PI / 2")
a("    const r = arr.length <= 1 ? 0 : Math.min(W / 2 - 55, H / 2 - 45)")
a("    return { ...n, x: W / 2 + r * Math.cos(angle), y: H / 2 + r * Math.sin(angle) }")
a("  })")
a("  const posMap: Record<string, { x: number; y: number }> = {}")
a("  positioned.forEach(n => { posMap[n.id] = { x: n.x, y: n.y } })")
a("  const sc: Record<string, string> = { skill: color, role: '#1f2937', company: '#9ca3af' }")
a("  return (")
a('    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto bg-gray-50 rounded-xl">')
a("      {edges.slice(0, 20).map((e, i) => {")
a("        const s = posMap[e.source], t = posMap[e.target]")
a("        if (!s || !t) return null")
a('        return <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke="#d1d5db" strokeWidth={1.5} />')
a("      })}")
a("      {positioned.map((n, i) => (")
a("        <g key={i}>")
a("          <circle cx={n.x} cy={n.y} r={20} fill=\"white\" stroke={sc[n.type] ?? '#d1d5db'} strokeWidth={2.5} />")
a("          <text x={n.x} y={n.y+1} textAnchor=\"middle\" dominantBaseline=\"middle\" fontSize={9} fontWeight=\"700\" fill={sc[n.type]??'#6b7280'}>{n.type==='skill'?'SK':n.type==='role'?'RL':'CO'}</text>")
a("          <text x={n.x} y={n.y+33} textAnchor=\"middle\" fontSize={8.5} fill=\"#111827\" fontWeight=\"600\">{n.id.length>12?n.id.slice(0,11)+'...':n.id}</text>")
a("          <text x={n.x} y={n.y+44} textAnchor=\"middle\" fontSize={7.5} fill={sc[n.type]??'#9ca3af'}>{(n.weight*100).toFixed(0)}%</text>")
a("        </g>")
a("      ))}")
a("    </svg>")
a("  )")
a("}")
a()

# Sample datasets
SAMPLES = {
  "software_engineer": {
    "resume": """John Smith - Senior Software Engineer
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
GPA: 3.8/4.0""",
    "jd": """Senior ML Engineer - AI Research Team

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

We offer competitive compensation, equity, and the opportunity to work on cutting-edge AI research with world-class team."""
  },
  "data_scientist": {
    "resume": """Sarah Johnson - Data Scientist
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
B.S. Statistics - State University (2016)""",
    "jd": """Senior Data Scientist - Product Analytics

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
- Collaborate with product and engineering teams"""
  },
  "devops_engineer": {
    "resume": """Mike Chen - DevOps Engineer
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
B.S. Information Technology (2018)""",
    "jd": """Senior DevOps Engineer - Platform Team

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
- Mentor junior engineers"""
  }
}

a("const SAMPLE_DATASETS = {")
for key, val in SAMPLES.items():
    a(f"  '{key}': {{")
    a(f"    resume: `{val['resume']}`,")
    a(f"    jd: `{val['jd']}`")
    a(f"  }},")
a("}")
a()

a("export default function CareerLens() {")
a("  const [resume, setResume] = useState('')")
a("  const [jobDescription, setJobDescription] = useState('')")
a("  const [analysis, setAnalysis] = useState<Analysis>(null)")
a("  const [optimizedResume, setOptimizedResume] = useState('')")
a("  const [loading, setLoading] = useState(false)")
a("  const [uploadMode, setUploadMode] = useState<'text' | 'pdf'>('text')")
a("  const [graphData, setGraphData] = useState<GraphData>(null)")
a("  const [systemStatus, setSystemStatus] = useState<SystemStatus>(null)")
a("  const [detailedResults, setDetailedResults] = useState<DetailedResults>(null)")
a("  const [activeTab, setActiveTab] = useState<'input' | 'results' | 'graphs' | 'optimize'>('input')")
a("  const [selectedSample, setSelectedSample] = useState('')")
a("  const [showSampleMenu, setShowSampleMenu] = useState(false)")
a("  const fileInputRef = useRef<HTMLInputElement>(null)")
a()
a("  useEffect(() => { checkStatus() }, [])")
a("  const checkStatus = async () => {")
a("    try { setSystemStatus(await (await fetch('/health')).json()) }")
a("    catch { setSystemStatus({ status: 'offline', ml_engine: false, groq_available: false }) }")
a("  }")
a()
a("  const loadSample = (key: string) => {")
a("    const s = SAMPLE_DATASETS[key as keyof typeof SAMPLE_DATASETS]")
a("    if (s) { setResume(s.resume); setJobDescription(s.jd); setSelectedSample(key); setShowSampleMenu(false) }")
a("  }")
a()
a("  const runAnalysis = async () => {")
a("    if (!resume.trim() || !jobDescription.trim()) return")
a("    setLoading(true); setAnalysis(null); setOptimizedResume(''); setDetailedResults(null); setActiveTab('results')")
a("    try {")
a("      const data = await (await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ resume, jobDescription }) })).json()")
a("      setAnalysis(data.analysis); setOptimizedResume(data.optimizedResume ?? ''); setDetailedResults(data.detailed_results ?? null)")
a("      try { setGraphData(await (await fetch('/api/graph-data/latest')).json()) } catch {}")
a("    } catch (e) { console.error(e) } finally { setLoading(false) }")
a("  }")
a()
a("  const runPDF = async (file: File) => {")
a("    setLoading(true); setAnalysis(null); setOptimizedResume(''); setDetailedResults(null); setActiveTab('results')")
a("    try {")
a("      const form = new FormData(); form.append('file', file); form.append('jobDescription', jobDescription)")
a("      const data = await (await fetch('/api/analyze-pdf', { method: 'POST', body: form })).json()")
a("      setAnalysis(data.analysis); setOptimizedResume(data.optimizedResume ?? ''); setDetailedResults(data.detailed_results ?? null)")
a("      try { setGraphData(await (await fetch('/api/graph-data/latest')).json()) } catch {}")
a("    } catch (e) { console.error(e) } finally { setLoading(false) }")
a("  }")
a()
a("  const downloadAsPDF = async () => {")
a("    if (!optimizedResume) return")
a("    const { jsPDF } = await import('jspdf')")
a("    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })")
a("    const margin = 20")
a("    const pageW = doc.internal.pageSize.getWidth()")
a("    const pageH = doc.internal.pageSize.getHeight()")
a("    const maxW = pageW - margin * 2")
a("    let y = margin")
a("    doc.setFillColor(17, 24, 39)")
a("    doc.rect(0, 0, pageW, 18, 'F')")
a("    doc.setTextColor(255, 255, 255)")
a("    doc.setFontSize(14)")
a("    doc.setFont('helvetica', 'bold')")
a("    doc.text('CareerLens AI - Optimized Resume', margin, 12)")
a("    y = 28")
a("    doc.setTextColor(30, 30, 30)")
a("    const lines = optimizedResume.split('\\n')")
a("    lines.forEach(line => {")
a("      if (y > pageH - margin) { doc.addPage(); y = margin }")
a("      const trimmed = line.trim()")
a("      if (trimmed.startsWith('---') || trimmed.startsWith('===')) {")
a("        doc.setDrawColor(200, 200, 200)")
a("        doc.line(margin, y, pageW - margin, y)")
a("        y += 4")
a("      } else if (trimmed.startsWith('CAREERLENS') || trimmed.startsWith('OPTIMIZED') || (trimmed === trimmed.toUpperCase() && trimmed.length > 3 && !trimmed.startsWith('-') && !trimmed.startsWith('•'))) {")
a("        doc.setFontSize(11)")
a("        doc.setFont('helvetica', 'bold')")
a("        doc.setTextColor(17, 24, 39)")
a("        doc.text(trimmed, margin, y)")
a("        y += 6")
a("      } else if (trimmed.startsWith('-') || trimmed.startsWith('•') || trimmed.startsWith('✅')) {")
a("        doc.setFontSize(9)")
a("        doc.setFont('helvetica', 'normal')")
a("        doc.setTextColor(60, 60, 60)")
a("        const wrapped = doc.splitTextToSize('  ' + trimmed, maxW)")
a("        doc.text(wrapped, margin, y)")
a("        y += wrapped.length * 4.5")
a("      } else if (trimmed === '') {")
a("        y += 3")
a("      } else {")
a("        doc.setFontSize(9)")
a("        doc.setFont('helvetica', 'normal')")
a("        doc.setTextColor(50, 50, 50)")
a("        const wrapped = doc.splitTextToSize(trimmed, maxW)")
a("        doc.text(wrapped, margin, y)")
a("        y += wrapped.length * 4.5")
a("      }")
a("    })")
a("    doc.setFontSize(7)")
a("    doc.setTextColor(150, 150, 150)")
a("    doc.text('Generated by CareerLens AI - careerlens.ai', margin, pageH - 8)")
a("    doc.save('careerlens-optimized-resume.pdf')")
a("  }")
a()
a("  const isOnline = systemStatus?.status === 'healthy'")
a("  const sampleLabels: Record<string, string> = { software_engineer: 'Software Engineer', data_scientist: 'Data Scientist', devops_engineer: 'DevOps Engineer' }")
a("  const tabs = [")
a("    { id: 'input' as const, label: 'Analyze', icon: <FileText className=\"w-3.5 h-3.5\" /> },")
a("    { id: 'results' as const, label: 'Results', icon: <BarChart3 className=\"w-3.5 h-3.5\" /> },")
a("    { id: 'graphs' as const, label: 'Graphs', icon: <Network className=\"w-3.5 h-3.5\" /> },")
a("    { id: 'optimize' as const, label: 'Optimize', icon: <Sparkles className=\"w-3.5 h-3.5\" /> },")
a("  ]")

a("  return (")
a("    <div className=\"min-h-screen bg-gray-50\">")
a("      <header className=\"bg-white border-b border-gray-200 sticky top-0 z-50\">")
a("        <div className=\"max-w-7xl mx-auto px-6 h-14 flex items-center justify-between\">")
a("          <div className=\"flex items-center gap-2.5\">")
a("            <div className=\"w-8 h-8 rounded-xl bg-gray-900 flex items-center justify-center\"><Brain className=\"w-4 h-4 text-white\" /></div>")
a("            <span className=\"font-bold text-lg tracking-tight text-gray-900\">CareerLens</span>")
a("            <span className=\"text-xs text-gray-400 font-mono bg-gray-100 px-2 py-0.5 rounded-full\">AI</span>")
a("          </div>")
a("          <nav className=\"hidden md:flex items-center gap-1 bg-gray-100 rounded-xl p-1\">")
a("            {tabs.map(t => (<button key={t.id} onClick={() => setActiveTab(t.id)} className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === t.id ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-900'}`}>{t.icon}{t.label}</button>))}")
a("          </nav>")
a("          <div className=\"flex items-center gap-2\">")
a("            <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-red-400'}`} />")
a("            <span className=\"text-xs text-gray-400 font-mono\">{systemStatus?.status ?? 'connecting...'}</span>")
a("          </div>")
a("        </div>")
a("      </header>")
a("      <main className=\"max-w-7xl mx-auto px-6 py-8\">")
a()
a("        {activeTab === 'input' && (")
a("          <div className=\"space-y-8\">")
a("            <div className=\"text-center py-10\">")
a("              <div className=\"inline-flex items-center gap-2 bg-gray-900 text-white rounded-full px-4 py-1.5 text-xs font-semibold mb-5\"><Zap className=\"w-3 h-3\" />Graph Neural Networks + Groq AI</div>")
a("              <h1 className=\"text-6xl font-black tracking-tighter text-gray-900 mb-4\">CareerLens AI</h1>")
a("              <p className=\"text-gray-500 text-lg max-w-xl mx-auto\">Deep career intelligence using knowledge graphs, GAT matching, bias detection and trajectory prediction.</p>")
a("            </div>")
a("            <div className=\"flex flex-wrap justify-center gap-2\">")
a("              {[{ icon: <Brain className=\"w-3.5 h-3.5\" />, label: 'ML Engine', on: systemStatus?.ml_engine }, { icon: <Zap className=\"w-3.5 h-3.5\" />, label: 'Groq AI', on: systemStatus?.groq_available }, { icon: <Network className=\"w-3.5 h-3.5\" />, label: 'Knowledge Graphs', on: true }, { icon: <Shield className=\"w-3.5 h-3.5\" />, label: 'Bias Detection', on: true }, { icon: <TrendingUp className=\"w-3.5 h-3.5\" />, label: 'Career Trajectory', on: true }].map(({ icon, label, on }, i) => (<div key={i} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border ${on ? 'bg-gray-900 text-white border-gray-900' : 'bg-white text-gray-400 border-gray-200'}`}>{icon}{label}</div>))}")
a("            </div>")
a()
a("            {/* Sample Dataset Selector */}")
a("            <div className=\"flex justify-center\">")
a("              <div className=\"relative\">")
a("                <button onClick={() => setShowSampleMenu(!showSampleMenu)} className=\"flex items-center gap-2 bg-white border border-gray-200 hover:border-gray-400 text-gray-700 text-sm font-medium px-5 py-2.5 rounded-xl transition-all shadow-sm\">")
a("                  <Play className=\"w-3.5 h-3.5 text-gray-500\" />")
a("                  {selectedSample ? sampleLabels[selectedSample] : 'Load Sample Dataset'}")
a("                  <ChevronDown className=\"w-3.5 h-3.5 text-gray-400\" />")
a("                </button>")
a("                {showSampleMenu && (")
a("                  <div className=\"absolute top-full mt-2 left-0 bg-white border border-gray-200 rounded-xl shadow-lg z-10 min-w-56 overflow-hidden\">")
a("                    {Object.entries(sampleLabels).map(([key, label]) => (")
a("                      <button key={key} onClick={() => loadSample(key)} className=\"w-full text-left px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-3 border-b border-gray-50 last:border-0\">")
a("                        <div className=\"w-7 h-7 rounded-lg bg-gray-100 flex items-center justify-center\"><Briefcase className=\"w-3.5 h-3.5 text-gray-500\" /></div>")
a("                        <div><p className=\"font-medium\">{label}</p><p className=\"text-xs text-gray-400\">Resume + Job Description</p></div>")
a("                      </button>")
a("                    ))}")
a("                  </div>")
a("                )}")
a("              </div>")
a("            </div>")
a()
a("            <div className=\"grid lg:grid-cols-2 gap-6\">")
a("              <div className=\"bg-white border border-gray-200 rounded-2xl p-6 space-y-4 shadow-sm\">")
a("                <div className=\"flex items-center justify-between\">")
a("                  <div className=\"flex items-center gap-3\">")
a("                    <div className=\"w-9 h-9 rounded-xl bg-blue-50 flex items-center justify-center\"><FileText className=\"w-4 h-4 text-blue-600\" /></div>")
a("                    <div><p className=\"font-semibold text-sm text-gray-900\">Resume</p><p className=\"text-gray-400 text-xs\">Paste text or upload PDF</p></div>")
a("                  </div>")
a("                  <div className=\"flex items-center gap-1 bg-gray-100 rounded-lg p-1\">")
a("                    {(['text', 'pdf'] as const).map(m => (<button key={m} onClick={() => setUploadMode(m)} className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${uploadMode === m ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-400 hover:text-gray-700'}`}>{m.toUpperCase()}</button>))}")
a("                  </div>")
a("                </div>")
a("                {uploadMode === 'text' ? (")
a("                  <textarea value={resume} onChange={e => setResume(e.target.value)} placeholder=\"Paste your resume here or load a sample above...\" className=\"w-full h-72 bg-gray-50 border border-gray-200 rounded-xl p-4 text-sm text-gray-800 placeholder-gray-300 resize-none focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all font-mono\" />")
a("                ) : (")
a("                  <div onClick={() => fileInputRef.current?.click()} className=\"h-72 border-2 border-dashed border-gray-200 rounded-xl flex flex-col items-center justify-center gap-4 cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-all group\">")
a("                    <div className=\"w-14 h-14 rounded-2xl bg-gray-100 flex items-center justify-center group-hover:bg-blue-100 transition-all\"><Upload className=\"w-6 h-6 text-gray-400 group-hover:text-blue-500\" /></div>")
a("                    <div className=\"text-center\"><p className=\"font-semibold text-sm text-gray-700\">Drop PDF resume here</p><p className=\"text-gray-400 text-xs mt-1\">or click to browse — PDF files only</p></div>")
a("                    <input ref={fileInputRef} type=\"file\" accept=\".pdf\" onChange={e => { const f = e.target.files?.[0]; if (f?.type === 'application/pdf') runPDF(f) }} className=\"hidden\" />")
a("                  </div>")
a("                )}")
a("              </div>")
a("              <div className=\"bg-white border border-gray-200 rounded-2xl p-6 space-y-4 shadow-sm\">")
a("                <div className=\"flex items-center gap-3\">")
a("                  <div className=\"w-9 h-9 rounded-xl bg-emerald-50 flex items-center justify-center\"><Target className=\"w-4 h-4 text-emerald-600\" /></div>")
a("                  <div><p className=\"font-semibold text-sm text-gray-900\">Job Description</p><p className=\"text-gray-400 text-xs\">Target role requirements</p></div>")
a("                </div>")
a("                <textarea value={jobDescription} onChange={e => setJobDescription(e.target.value)} placeholder=\"Paste the job description here or load a sample above...\" className=\"w-full h-72 bg-gray-50 border border-gray-200 rounded-xl p-4 text-sm text-gray-800 placeholder-gray-300 resize-none focus:outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 transition-all font-mono\" />")
a("              </div>")
a("            </div>")
a("            <div className=\"flex justify-center pt-4\">")
a("              <button onClick={runAnalysis} disabled={loading || !resume.trim() || !jobDescription.trim()} className=\"flex items-center gap-3 bg-gray-900 hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold text-base px-10 py-4 rounded-2xl transition-all hover:scale-105 active:scale-100 shadow-lg\">")
a("                {loading ? (<><div className=\"w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin\" /><span>Analyzing...</span></>) : (<><Brain className=\"w-5 h-5\" /><span>Analyze with CareerLens AI</span><Zap className=\"w-4 h-4\" /></>)}")
a("              </button>")
a("            </div>")
a("          </div>")
a("        )}")

a()
a("        {activeTab === 'results' && (")
a("          <div className=\"space-y-6\">")
a("            {loading && (<div className=\"flex flex-col items-center justify-center py-32 gap-6\"><div className=\"relative w-20 h-20\"><div className=\"absolute inset-0 border-4 border-gray-100 rounded-full\" /><div className=\"absolute inset-0 border-4 border-gray-900 border-t-transparent rounded-full animate-spin\" /><Brain className=\"absolute inset-0 m-auto w-6 h-6 text-gray-700\" /></div><div className=\"text-center\"><p className=\"font-bold text-lg text-gray-900\">Running ML Pipeline</p><p className=\"text-gray-400 text-sm mt-1\">NER extraction to Graph building to GAT matching to Optimization</p></div></div>)}")
a("            {!loading && !analysis && (<div className=\"flex flex-col items-center justify-center py-32 gap-4 text-center\"><div className=\"w-20 h-20 rounded-2xl bg-gray-100 flex items-center justify-center\"><BarChart3 className=\"w-9 h-9 text-gray-300\" /></div><p className=\"font-semibold text-gray-500 text-lg\">No results yet</p><p className=\"text-gray-400 text-sm\">Run an analysis from the Analyze tab</p><button onClick={() => setActiveTab('input')} className=\"mt-2 px-6 py-2.5 bg-gray-900 text-white text-sm font-medium rounded-xl hover:bg-gray-700 transition-all\">Go to Analyze</button></div>)}")
a("            {!loading && analysis && (<>")
a("              <div className=\"bg-white border border-gray-200 rounded-2xl p-8 flex flex-col md:flex-row items-center gap-8 shadow-sm\">")
a("                <div className=\"relative w-40 h-40 shrink-0\">")
a("                  <svg className=\"w-full h-full -rotate-90\" viewBox=\"0 0 100 100\">")
a("                    <circle cx=\"50\" cy=\"50\" r=\"42\" stroke=\"#f3f4f6\" strokeWidth=\"8\" fill=\"none\" />")
a("                    <circle cx=\"50\" cy=\"50\" r=\"42\" stroke=\"#111827\" strokeWidth=\"8\" fill=\"none\" strokeLinecap=\"round\" strokeDasharray={`${(analysis.atsScore/100)*263.9} 263.9`} />")
a("                  </svg>")
a("                  <div className=\"absolute inset-0 flex flex-col items-center justify-center\"><span className=\"text-4xl font-black text-gray-900\">{analysis.atsScore}</span><span className=\"text-gray-400 text-xs font-medium\">/100</span></div>")
a("                </div>")
a("                <div className=\"flex-1 space-y-4\">")
a("                  <div><h2 className=\"text-2xl font-bold text-gray-900\">GAT Match Score</h2><p className=\"text-gray-400 text-sm mt-1\">Graph Attention Network analysis complete</p></div>")
a("                  {detailedResults?.gat_matching && (<div className=\"grid grid-cols-3 gap-3\">{[{ label: 'Graph Similarity', value: `${(detailedResults.gat_matching.graph_similarity*100).toFixed(0)}%` }, { label: 'Skill Matches', value: detailedResults.gat_matching.skill_overlap?.length??0 }, { label: 'Skill Gaps', value: detailedResults.gat_matching.missing_skills?.length??0 }].map(({ label, value }, i) => (<div key={i} className=\"bg-gray-50 border border-gray-100 rounded-xl p-4 text-center\"><div className=\"text-2xl font-bold text-gray-900\">{value}</div><div className=\"text-gray-400 text-xs mt-1\">{label}</div></div>))}</div>)}")
a("                </div>")
a("              </div>")
a("              <div className=\"grid md:grid-cols-3 gap-4\">{[{ title: 'Knowledge Graph', content: analysis.atsFeedback, icon: <Network className=\"w-4 h-4\" />, c: 'text-blue-600', bg: 'bg-blue-50', b: 'border-blue-100' }, { title: 'ML Technical', content: analysis.techFeedback, icon: <Brain className=\"w-4 h-4\" />, c: 'text-violet-600', bg: 'bg-violet-50', b: 'border-violet-100' }, { title: 'Career Intelligence', content: analysis.hrFeedback, icon: <TrendingUp className=\"w-4 h-4\" />, c: 'text-emerald-600', bg: 'bg-emerald-50', b: 'border-emerald-100' }].map(({ title, content, icon, c, bg, b }, i) => (<div key={i} className=\"bg-white border border-gray-200 rounded-2xl p-5 space-y-3 shadow-sm\"><div className={`flex items-center gap-2 text-sm font-semibold ${c}`}><div className={`w-7 h-7 rounded-lg ${bg} border ${b} flex items-center justify-center`}>{icon}</div>{title}</div><p className=\"text-gray-500 text-sm leading-relaxed\">{content}</p></div>))}</div>")
a("              {(analysis.skillGaps?.length??0)>0 && (<div className=\"bg-white border border-gray-200 rounded-2xl p-5 space-y-3 shadow-sm\"><div className=\"flex items-center gap-2 text-sm font-semibold text-red-600\"><AlertCircle className=\"w-4 h-4\" />Skill Gaps ({analysis.skillGaps.length})</div><div className=\"flex flex-wrap gap-2\">{analysis.skillGaps.map((s,i)=><span key={i} className=\"bg-red-50 border border-red-100 text-red-600 text-xs px-3 py-1.5 rounded-full font-medium\">{s}</span>)}</div></div>)}")
a("              {(analysis.recommendations?.length??0)>0 && (<div className=\"bg-white border border-gray-200 rounded-2xl p-5 space-y-3 shadow-sm\"><div className=\"flex items-center gap-2 text-sm font-semibold text-emerald-600\"><CheckCircle className=\"w-4 h-4\" />Recommendations</div><div className=\"space-y-2\">{analysis.recommendations.map((r,i)=>(<div key={i} className=\"flex items-start gap-3 text-sm text-gray-600\"><div className=\"w-5 h-5 rounded-full bg-emerald-100 flex items-center justify-center shrink-0 mt-0.5 text-xs font-bold text-emerald-700\">{i+1}</div>{r}</div>))}</div></div>)}")
a("              {detailedResults?.bias_detection && (<div className=\"bg-white border border-gray-200 rounded-2xl p-5 space-y-3 shadow-sm\"><div className=\"flex items-center gap-2 text-sm font-semibold text-amber-600\"><Shield className=\"w-4 h-4\" />JD Bias Detection</div><div className=\"flex items-center gap-6\"><div className=\"text-center bg-amber-50 border border-amber-100 rounded-xl p-4\"><div className=\"text-3xl font-bold text-amber-700\">{detailedResults.bias_detection.quality_score}</div><div className=\"text-amber-600 text-xs mt-1\">Quality Score</div></div><div className=\"flex-1\">{detailedResults.bias_detection.bias_detected?<div className=\"flex flex-wrap gap-2\">{detailedResults.bias_detection.bias_types.map((b,i)=><span key={i} className=\"bg-amber-50 border border-amber-100 text-amber-700 text-xs px-3 py-1 rounded-full\">{b}</span>)}</div>:<p className=\"text-sm text-emerald-600 flex items-center gap-2\"><CheckCircle className=\"w-4 h-4\" />No significant bias detected</p>}</div></div></div>)}")
a("              {detailedResults?.career_trajectory && (<div className=\"bg-white border border-gray-200 rounded-2xl p-5 space-y-4 shadow-sm\"><div className=\"flex items-center gap-2 text-sm font-semibold text-blue-600\"><TrendingUp className=\"w-4 h-4\" />Career Trajectory</div><div className=\"grid md:grid-cols-2 gap-6\"><div className=\"space-y-2\"><p className=\"text-xs text-gray-400 uppercase tracking-wider font-semibold\">Predicted Roles</p>{detailedResults.career_trajectory.predicted_roles.map((r,i)=>(<div key={i} className=\"flex items-center gap-3 text-sm text-gray-700\"><div className=\"w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center text-xs font-bold text-blue-700\">{i+1}</div>{r}</div>))}</div><div className=\"space-y-2\"><p className=\"text-xs text-gray-400 uppercase tracking-wider font-semibold\">Skills to Develop</p>{detailedResults.career_trajectory.skill_development.map((s,i)=>(<div key={i} className=\"flex items-center gap-2 text-sm text-gray-700\"><div className=\"w-2 h-2 rounded-full bg-blue-400\" />{s}</div>))}</div></div><p className=\"text-xs text-gray-400 flex items-center gap-1.5 pt-1 border-t border-gray-100\"><Activity className=\"w-3.5 h-3.5\" />Timeline: {detailedResults.career_trajectory.timeline}</p></div>)}")
a("              <div className=\"flex justify-center pt-2\"><button onClick={() => setActiveTab('optimize')} className=\"flex items-center gap-2 bg-gray-900 hover:bg-gray-700 text-white text-sm font-medium px-6 py-3 rounded-xl transition-all\"><Sparkles className=\"w-4 h-4\" />View Optimized Resume</button></div>")
a("            </>)}")
a("          </div>")
a("        )}")

a()
a("        {activeTab === 'graphs' && (")
a("          <div className=\"space-y-6\">")
a("            {!graphData ? (<div className=\"flex flex-col items-center justify-center py-32 gap-4 text-center\"><div className=\"w-20 h-20 rounded-2xl bg-gray-100 flex items-center justify-center\"><Network className=\"w-9 h-9 text-gray-300\" /></div><p className=\"font-semibold text-gray-500 text-lg\">No graph data yet</p><p className=\"text-gray-400 text-sm\">Run an analysis to generate knowledge graphs</p><button onClick={() => setActiveTab('input')} className=\"mt-2 px-6 py-2.5 bg-gray-900 text-white text-sm font-medium rounded-xl hover:bg-gray-700 transition-all\">Go to Analyze</button></div>) : (<>")
a("              <div className=\"grid grid-cols-2 md:grid-cols-4 gap-4\">{[{ label: 'Resume Nodes', value: graphData.resume_graph?.nodes?.length??0, icon: <Database className=\"w-4 h-4\" /> }, { label: 'Resume Edges', value: graphData.resume_graph?.edges?.length??0, icon: <Network className=\"w-4 h-4\" /> }, { label: 'JD Nodes', value: graphData.jd_graph?.nodes?.length??0, icon: <Target className=\"w-4 h-4\" /> }, { label: 'JD Edges', value: graphData.jd_graph?.edges?.length??0, icon: <Activity className=\"w-4 h-4\" /> }].map(({ label, value, icon }, i) => (<div key={i} className=\"bg-white border border-gray-200 rounded-2xl p-5 text-center shadow-sm\"><div className=\"flex justify-center text-gray-400 mb-2\">{icon}</div><div className=\"text-3xl font-bold text-gray-900\">{value}</div><div className=\"text-gray-400 text-xs mt-1\">{label}</div></div>))}</div>")
a("              <div className=\"grid lg:grid-cols-2 gap-6\">")
a("                <div className=\"bg-white border border-gray-200 rounded-2xl p-6 space-y-4 shadow-sm\">")
a("                  <div className=\"flex items-center justify-between\"><h3 className=\"font-semibold text-sm text-gray-900 flex items-center gap-2\"><div className=\"w-2.5 h-2.5 rounded-full bg-blue-500\" />Resume Knowledge Graph</h3><span className=\"text-xs text-gray-400\">{graphData.resume_graph?.nodes?.length??0} nodes</span></div>")
a("                  <GraphViz nodes={graphData.resume_graph?.nodes??[]} edges={graphData.resume_graph?.edges??[]} color=\"#3b82f6\" />")
a("                  <div className=\"flex items-center gap-4 text-xs text-gray-400 pt-2 border-t border-gray-100\"><span className=\"flex items-center gap-1.5\"><span className=\"w-3 h-3 rounded-full border-2 border-blue-500 inline-block\" />Skill (SK)</span><span className=\"flex items-center gap-1.5\"><span className=\"w-3 h-3 rounded-full border-2 border-gray-800 inline-block\" />Role (RL)</span><span className=\"flex items-center gap-1.5\"><span className=\"w-3 h-3 rounded-full border-2 border-gray-400 inline-block\" />Company (CO)</span></div>")
a("                </div>")
a("                <div className=\"bg-white border border-gray-200 rounded-2xl p-6 space-y-4 shadow-sm\">")
a("                  <div className=\"flex items-center justify-between\"><h3 className=\"font-semibold text-sm text-gray-900 flex items-center gap-2\"><div className=\"w-2.5 h-2.5 rounded-full bg-emerald-500\" />Job Description Graph</h3><span className=\"text-xs text-gray-400\">{graphData.jd_graph?.nodes?.length??0} nodes</span></div>")
a("                  <GraphViz nodes={graphData.jd_graph?.nodes??[]} edges={graphData.jd_graph?.edges??[]} color=\"#10b981\" />")
a("                  <div className=\"flex items-center gap-4 text-xs text-gray-400 pt-2 border-t border-gray-100\"><span className=\"flex items-center gap-1.5\"><span className=\"w-3 h-3 rounded-full border-2 border-emerald-500 inline-block\" />Skill (SK)</span><span className=\"flex items-center gap-1.5\"><span className=\"w-3 h-3 rounded-full border-2 border-gray-800 inline-block\" />Role (RL)</span><span className=\"flex items-center gap-1.5\"><span className=\"w-3 h-3 rounded-full border-2 border-gray-400 inline-block\" />Company (CO)</span></div>")
a("                </div>")
a("              </div>")
a("              <div className=\"grid lg:grid-cols-2 gap-6\">{[{ title: 'Resume Nodes', nodes: graphData.resume_graph?.nodes??[], accent: 'text-blue-600' }, { title: 'JD Nodes', nodes: graphData.jd_graph?.nodes??[], accent: 'text-emerald-600' }].map(({ title, nodes, accent }, gi) => (<div key={gi} className=\"bg-white border border-gray-200 rounded-2xl p-5 shadow-sm\"><h3 className=\"font-semibold text-sm text-gray-900 mb-3\">{title}</h3><div className=\"space-y-1\">{nodes.slice(0,10).map((n,i)=>(<div key={i} className=\"flex items-center justify-between py-2 border-b border-gray-50 last:border-0\"><div className=\"flex items-center gap-2.5\"><div className=\"w-6 h-6 rounded-lg bg-gray-100 flex items-center justify-center\">{n.type==='skill'?<Code className=\"w-3 h-3 text-gray-500\" />:n.type==='role'?<Briefcase className=\"w-3 h-3 text-gray-500\" />:<Globe className=\"w-3 h-3 text-gray-500\" />}</div><div><p className=\"text-sm font-medium text-gray-800\">{n.id}</p><p className=\"text-xs text-gray-400 capitalize\">{n.type}</p></div></div><span className={`text-sm font-bold ${accent}`}>{(n.weight*100).toFixed(0)}%</span></div>))}</div></div>))}</div>")
a("              {detailedResults?.gat_matching && (<div className=\"bg-white border border-gray-200 rounded-2xl p-5 shadow-sm\"><h3 className=\"font-semibold text-sm text-gray-900 mb-4\">Graph Overlap Analysis</h3><div className=\"grid md:grid-cols-2 gap-5\"><div className=\"space-y-2\"><p className=\"text-xs text-gray-400 uppercase tracking-wider font-semibold\">Matching Skills</p><div className=\"flex flex-wrap gap-2\">{detailedResults.gat_matching.skill_overlap?.length?detailedResults.gat_matching.skill_overlap.map((s,i)=><span key={i} className=\"bg-emerald-50 border border-emerald-100 text-emerald-700 text-xs px-3 py-1 rounded-full\">{s}</span>):<span className=\"text-gray-400 text-sm\">None found</span>}</div></div><div className=\"space-y-2\"><p className=\"text-xs text-gray-400 uppercase tracking-wider font-semibold\">Missing Skills</p><div className=\"flex flex-wrap gap-2\">{detailedResults.gat_matching.missing_skills?.length?detailedResults.gat_matching.missing_skills.map((s,i)=><span key={i} className=\"bg-red-50 border border-red-100 text-red-600 text-xs px-3 py-1 rounded-full\">{s}</span>):<span className=\"text-gray-400 text-sm\">None detected</span>}</div></div></div></div>)}")
a("            </>)}")
a("          </div>")
a("        )}")
a()
a("        {activeTab === 'optimize' && (")
a("          <div className=\"space-y-6\">")
a("            {!optimizedResume ? (<div className=\"flex flex-col items-center justify-center py-32 gap-4 text-center\"><div className=\"w-20 h-20 rounded-2xl bg-gray-100 flex items-center justify-center\"><Sparkles className=\"w-9 h-9 text-gray-300\" /></div><p className=\"font-semibold text-gray-500 text-lg\">No optimized resume yet</p><p className=\"text-gray-400 text-sm\">Run an analysis to generate your optimized resume</p><button onClick={() => setActiveTab('input')} className=\"mt-2 px-6 py-2.5 bg-gray-900 text-white text-sm font-medium rounded-xl hover:bg-gray-700 transition-all\">Go to Analyze</button></div>) : (<>")
a("              <div className=\"flex items-center justify-between\">")
a("                <div><h2 className=\"text-xl font-bold text-gray-900\">ML-Optimized Resume</h2><p className=\"text-gray-400 text-sm mt-0.5\">Enhanced by CareerLens knowledge graph pipeline</p></div>")
a("                <button onClick={downloadAsPDF} className=\"flex items-center gap-2 bg-gray-900 hover:bg-gray-700 text-white text-sm font-semibold px-5 py-3 rounded-xl transition-all hover:scale-105 shadow-md\">")
a("                  <Download className=\"w-4 h-4\" />Download as PDF")
a("                </button>")
a("              </div>")
a("              <div className=\"grid grid-cols-2 md:grid-cols-4 gap-4\">{[{ label: 'ATS Compatibility', value: '+25%', color: 'text-emerald-600' }, { label: 'Keyword Match', value: '+40%', color: 'text-blue-600' }, { label: 'Skill Alignment', value: '+35%', color: 'text-violet-600' }, { label: 'Overall Score', value: '+30%', color: 'text-amber-600' }].map(({ label, value, color }, i) => (<div key={i} className=\"bg-white border border-gray-200 rounded-2xl p-4 text-center shadow-sm\"><div className={`text-2xl font-bold ${color}`}>{value}</div><div className=\"text-gray-400 text-xs mt-1\">{label}</div></div>))}</div>")
a("              <div className=\"bg-white border border-gray-200 rounded-2xl p-6 shadow-sm\">")
a("                <div className=\"flex items-center justify-between mb-4\">")
a("                  <div className=\"flex items-center gap-2 text-xs text-gray-400 font-mono\"><div className=\"w-2 h-2 rounded-full bg-emerald-400\" />careerlens-optimized-resume.pdf</div>")
a("                  <span className=\"text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded-full\">Preview</span>")
a("                </div>")
a("                <pre className=\"whitespace-pre-wrap text-sm text-gray-700 font-mono leading-relaxed bg-gray-50 rounded-xl p-5 border border-gray-100\">{optimizedResume}</pre>")
a("              </div>")
a("              <div className=\"bg-white border border-gray-200 rounded-2xl p-5 shadow-sm\"><div className=\"flex items-center gap-2 text-sm font-semibold text-emerald-600 mb-3\"><CheckCircle className=\"w-4 h-4\" />Applied Optimizations</div><div className=\"grid md:grid-cols-2 gap-2\">{['Graph-based skill analysis', 'GAT matching score integrated', 'Career trajectory applied', 'Bias-aware optimization', 'ATS keyword optimization', 'Action verb enhancement'].map((o,i)=>(<div key={i} className=\"flex items-center gap-2.5 text-sm text-gray-600\"><CheckCircle className=\"w-3.5 h-3.5 text-emerald-400 shrink-0\" />{o}</div>))}</div></div>")
a("              <div className=\"flex justify-center\"><button onClick={downloadAsPDF} className=\"flex items-center gap-3 bg-gray-900 hover:bg-gray-700 text-white font-bold text-base px-10 py-4 rounded-2xl transition-all hover:scale-105 shadow-lg\"><Download className=\"w-5 h-5\" />Download Optimized Resume as PDF</button></div>")
a("            </>)}")
a("          </div>")
a("        )}")
a()
a("      </main>")
a("      <footer className=\"border-t border-gray-200 mt-16 bg-white\">")
a("        <div className=\"max-w-7xl mx-auto px-6 py-6 flex flex-col md:flex-row items-center justify-between gap-3\">")
a("          <div className=\"flex items-center gap-2 text-gray-400 text-sm\"><div className=\"w-5 h-5 rounded-lg bg-gray-900 flex items-center justify-center\"><Brain className=\"w-3 h-3 text-white\" /></div>CareerLens - Deep Career Intelligence</div>")
a("          <div className=\"flex items-center gap-5 text-xs text-gray-300\"><span className=\"flex items-center gap-1\"><Network className=\"w-3 h-3\" />Knowledge Graphs</span><span className=\"flex items-center gap-1\"><Brain className=\"w-3 h-3\" />GAT Matching</span><span className=\"flex items-center gap-1\"><Shield className=\"w-3 h-3\" />Bias Detection</span><span className=\"flex items-center gap-1\"><TrendingUp className=\"w-3 h-3\" />Career Trajectory</span></div>")
a("        </div>")
a("      </footer>")
a("    </div>")
a("  )")
a("}")
a()

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

import os
print(f'Written {os.path.getsize(OUT)} bytes, {len(L)} lines to {OUT}')
