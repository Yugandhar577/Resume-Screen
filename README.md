# CareerLens

**AI-Powered Resume Transformation & Hiring Simulation Platform**

CareerLens simulates how resumes are evaluated in real hiring pipelines, providing transparent explanations and ethical resume optimization.

## 🎯 What Makes CareerLens Different

Unlike traditional ATS tools that only provide scores, CareerLens:

- **Simulates 3 Hiring Personas**: ATS System, Technical Lead, HR Recruiter
- **Provides Transparent Explanations**: Understand exactly why resumes get accepted/rejected
- **Ethical Optimization**: Never invents skills or experience - only enhances existing content
- **ATS-Safe Output**: Generates clean, downloadable, PDF-ready resumes

## 🏗️ Architecture

```
CareerLens/
├── frontend/           # Next.js React frontend
├── backend/           # FastAPI Python backend
├── ml/               # AI/ML components
│   ├── hiring_simulator.py    # Multi-persona simulation
│   ├── resume_optimizer.py    # Ethical resume transformation
│   ├── gat/                  # Graph Attention Networks
│   ├── bias/                 # Bias detection
│   └── rewriter/             # T5 fine-tuning
└── data/             # Data pipeline & scraping
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.9+
- Google Gemini API key

### Installation

1. **Clone and setup**:
```bash
git clone <repo>
cd careerlens
```

2. **Frontend setup**:
```bash
npm install
```

3. **Backend setup**:
```bash
pip install -r requirements.txt
```

4. **Environment variables**:
```bash
cp .env.example .env
# Add your GEMINI_API_KEY
```

### Running CareerLens

1. **Start backend**:
```bash
npm run backend
```

2. **Start frontend** (new terminal):
```bash
npm run dev
```

3. **Open**: http://localhost:3000

## 🧠 Core AI Components

### 1. Hiring Simulator (`ml/hiring_simulator.py`)
Simulates three distinct hiring personas:
- **ATS**: Keyword matching, format compliance
- **Technical Lead**: Technical depth, project quality
- **HR Recruiter**: Cultural fit, soft skills, career progression

### 2. Resume Optimizer (`ml/resume_optimizer.py`)
Ethically transforms resumes with strict constraints:
- Never invents skills or experience
- Only reorders, rephrases, emphasizes existing content
- Maintains candidate identity
- Ensures ATS compatibility

### 3. Advanced ML Pipeline
- **Graph Neural Networks**: Resume-JD matching via graph representation
- **Bias Detection**: Flags unrealistic requirements in job descriptions
- **T5 Fine-tuning**: Custom resume rewriting model

## 📊 API Endpoints

### `POST /api/analyze`
Core intelligence endpoint that:
1. Simulates hiring process with 3 personas
2. Generates optimized resume
3. Returns transparent analysis

```json
{
  "resume": "Your resume content...",
  "jobDescription": "Target job description..."
}
```

Response:
```json
{
  "analysis": {
    "atsScore": 85,
    "atsFeedback": "Strong keyword match...",
    "techFeedback": "Solid technical depth...",
    "hrFeedback": "Good career progression..."
  },
  "optimizedResume": "Optimized resume content..."
}
```

## 🎓 Academic Depth

CareerLens includes research-grade ML components:

- **Synthetic Hard Negatives**: Generate training data without real rejection labels
- **Graph Attention Networks**: Treat resumes as knowledge graphs
- **Contrastive Learning**: InfoNCE loss for resume-JD similarity
- **Knowledge Distillation**: GPT → T5 for custom rewriting models

## 🔒 Ethical AI Principles

1. **No Fake Content**: Never invents skills, experience, or achievements
2. **Transparency**: Clear explanations for all decisions
3. **Privacy**: No data storage, processing happens in real-time
4. **Fairness**: Bias detection in job descriptions

## 🚀 Deployment

### Production Build
```bash
npm run build
npm start
```

### Docker (Optional)
```bash
docker-compose up
```

## 📈 Roadmap

- [ ] PDF upload support
- [ ] Role-specific optimization templates
- [ ] Career trajectory prediction
- [ ] Integration with job boards
- [ ] Mobile app

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**CareerLens**: Turning resume optimization from guesswork into explainable AI-driven decisions.