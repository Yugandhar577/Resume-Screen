# Project structure

```text
app/                      Next.js UI (page.tsx is the maintained page)
lib/resume-pdf.mjs         Shared PDF export implementation
public/fonts/             Embedded/self-hosted Noto Sans and its license
main.py                   FastAPI API, validation, PDF parsing, optional AI orchestration
ml/careerlens_engine.py    Active deterministic evidence/coverage orchestration
ml/skills.py              Shared skill vocabulary, aliases, source-line evidence
ml/requirements.py        Required/preferred, negation, proficiency/degree/year uncertainty
ml/assessments.py         Distinct evidence-linked reviewer rubrics
ml/hiring_simulator.py     Optional opt-in Groq reviewer feedback
ml/resume_optimizer.py     Formatting-only cleanup and preservation validation
ml/rewrite_workflow.py     Reviewable constrained edits and section-template ordering
ml/pdf_processing.py       Local PDF/column/password extraction and optional OCR
ml/semantic_matching.py    Optional local, versioned MiniLM/text-graph/GAT inference
ml/model_registry.py       Checksum/provenance gates; trained != deployment-ready
integrations/              Public Arbeitnow adapter and pre-parser request/access guards
ml/ner/, graph/, gat/      Separate entity/graph/GAT research implementations
ml/bias/                  Separate BERT/weak-label quality research
ml/rewriter/              Separate T5 training/inference research
ml/trajectory/            Separate GraphSAGE career-transition research
data/                     Pinned public acquisition, real Snorkel aggregation, source utilities
training/                 GAT/GraphSAGE experiments and reviewed-data-gated T5/BERT paths
ml/models/*/manifest.json  Evaluation/provenance, not model weights (weights are ignored)
tests/                    API, engine, optional research, and PDF regression checks
FUTURE_IMPLEMENTATION_PROMPT.md   Remaining original product promises
STAGE_STATUS.md            Delivered stages, unavailable promises, remaining gates
DATA_SOURCES.md            Accepted/rejected sources, licenses, experiments and limitations
```

Runtime mapping:

1. The browser sends `/api/analyze` or multipart `/api/analyze-pdf` through Next.js.
2. Next.js proxies to `BACKEND_URL` (local port 8001; Docker `backend:8001`).
3. FastAPI validates inputs, extracts PDF text when needed, and invokes the same engine.
4. The engine returns coverage, requirements/reviewer evidence, per-request observed
   entity graphs, phrase flags, cleanup and reviewable edit proposals.
5. With explicit AI opt-in and provider configuration, Groq adds reviewer text;
   failure preserves baseline results with a warning.
6. Optional local semantic inference returns a separate experimental similarity and
   actual text-chunk attention. It never overwrites coverage or entity-graph weights.
7. Accepted edit IDs and a section template go to /api/optimize-preview; the server
   regenerates/verifies edits and reanalyzes the resulting resume against the same JD.
8. The UI displays that response and creates a clean, paginated PDF locally. Job-board
   import is separate and never receives candidate content.

No shared latest result or candidate-history database exists. GAT inference is opt-in;
career GraphSAGE is a real research artifact but explicitly deployment-disabled.
T5/BERT have reviewed-data training paths, not trained installed models. Capability
reporting makes these distinctions explicit. Public training data is never confused
with user upload retention.
