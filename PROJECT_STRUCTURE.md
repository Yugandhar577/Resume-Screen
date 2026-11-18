# Project structure

```text
app/                      Next.js UI (page.tsx is the maintained page)
ml/pdf_export.py          Active embedded-font PDF download, HarfBuzz shaping, Unicode mappings
lib/resume-pdf.mjs         Legacy client PDF helper and regression checks
public/fonts/             Embedded/self-hosted Noto Sans and its license
main.py                   FastAPI API, validation, PDF parsing, optional AI orchestration
ml/careerlens_engine.py    Active deterministic evidence/coverage orchestration
ml/skills.py              Shared skill vocabulary, aliases, source-line evidence
ml/requirements.py        Required/preferred, negation, proficiency/degree/year uncertainty
ml/assessments.py         Distinct evidence-linked reviewer rubrics
ml/local_review.py         Local Gemma evidence review and contextual qualifications
ml/local_rewriter.py       Protected facts, semantic comparison, signed review proposals
ml/hiring_simulator.py     Compatibility facade for local reviewers
ml/resume_optimizer.py     Formatting-only cleanup and preservation validation
ml/rewrite_workflow.py     Reviewable constrained edits and section-template ordering
ml/pdf_processing.py       Local PDF/column/password extraction and optional OCR
ml/semantic_matching.py    Local entity/source graph → MiniLM/GAT, attention, node-removal and synthetic calibration
ml/entity_graph.py         Pinned pretrained English NER + canonical spans, observed containment
ml/research_inference.py   Local BERT phrase sensitivity and trained T5 proposal candidates
ml/career_experiment.py    Explicit ESCO role mapping + historical GraphSAGE research explorer
ml/weak_wording.py         Fitted Snorkel numeric inference with abstentions
ml/windows_limits.py      Windows parser memory/CPU/process Job Object limits
ml/model_registry.py       Checksum/provenance gates; trained != deployment-ready
integrations/              Local Ollama transport, attributed Arbeitnow/Remote OK feeds, request/access guards
app/review/                AI-reviewed fictional examples, editable human decisions and restore/download
data/review_queue.py       Review type policy, explicit synthetic AI mode, checksum-bound exports
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
5. With local AI enabled, Ollama/Gemma adds source-checked reviewer text, qualification alignment, job wording review, entity extraction, and career exploration;
   failure preserves baseline results with a warning and never invokes a cloud provider.
6. Optional local semantic inference returns a separate experimental similarity and
   actual entity/source attention, node-removal sensitivity and synthetic benchmark calibration. It never overwrites coverage or entity-graph weights.
7. Accepted edit IDs and a section template go to /api/optimize-preview; the server
   regenerates/verifies edits and reanalyzes the resulting resume against the same JD.
8. The UI displays that response and requests a clean, paginated local PDF from /api/export-pdf. Job-board
   import is separate and never receives candidate content.

Local proposals are signed, expire after 30 minutes, and are bound to the exact resume/job.
PDF parsing uses a disposable process and a 45-second deadline.
No shared latest result or candidate-history database exists. GAT inference is opt-in;
career GraphSAGE is a real research artifact but explicitly deployment-disabled.
T5/BERT and Snorkel are trained locally on explicitly AI-reviewed fictional examples;
optional research controls never represent them as production-ready. Capability
reporting makes these distinctions explicit. Public training data is never confused
with user upload retention.

Optional BERT/Snorkel analysis uses useResearch. T5 proposals share the same fact/semantic/signature acceptance path as Gemma. The historical career explorer requires a user-selected ESCO role and does not infer candidate skills or suitability.


India-first operational additions:
- app/market-tools.tsx: sourced skill-based career matching and salary comparisons.
- ml/career_matching.py: exact evidence plus frozen-encoder retrieval over public ESCO profiles.
- data/acquire_career_profiles.py and training/build_career_index.py: resumable public acquisition and checksum-bound local index.
- ml/evidence_matching.py and app/evidence-matching.tsx: cross-occupation exact passage retrieval and qualifier review.
- ml/salary.py and data/india_salary_references.json: explicit offer units/basis and dated India reference ranges.
- SALARY_REFERENCES.md: coverage and user-supplied benchmark format.
- training/evaluate_operational.py: fresh authored invariant/negation stress evaluation.
