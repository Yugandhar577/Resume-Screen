# CareerLens

Local resume and job-description evidence review with a minimal black-and-white interface.
The web workflow uses Gemma through Ollama without provider credentials or cloud fallback.
Baseline analysis also works when the local model is unavailable.

## What works

- Paste text or upload a PDF; 5 MB/file, 30 pages, 50,000 characters per input.
- Canonical skill coverage, aliases, source evidence, basic negation, and required/preferred qualifications.
- Three evidence-linked reviewer perspectives: ATS/structure, technical evidence, and recruiter clarity.
- Optional local Gemma contextual qualification alignment, job wording review, entity extraction, and exploratory career ideas.
- Reviewable constrained edits plus optional local wording proposals. Local proposals undergo protected-fact checks plus either an exact approved-transformation certificate or a model comparison before being shown; neither check proves factual equivalence.
- Accept individual proposals, reorder existing sections using six role templates, and reanalyze actual coverage. Zero improvement is valid.
- Clean paginated local PDF download with embedded Noto Sans/Devanagari fonts, HarfBuzz shaping, and tested Unicode extraction for Latin, Greek, Cyrillic, and Hindi examples. UTF-8 text download and browser print are also available. PDF export is capped at 30 pages.
- Password handling, a two-column reading-order heuristic, optional local scanned-page OCR, and editable extracted text before reanalysis.
- Document parsing in a disposable process with a 45-second timeout. Linux applies CPU/address-space limits; Windows applies a 3 GB committed-memory limit, 40-second CPU limit, and one-process limit. These resource controls are not a complete hostile-document sandbox.
- Arbeitnow and Remote OK job-description import through supported public feeds. Listings retain source attribution and original source links; candidate text is never sent to either board.
- Optional local NER → observed entity/source graph → MiniLM → trained GAT matching, actual attention, and tested entity-removal sensitivity, separate from deterministic coverage. Weights and encoder have been reproduced on this machine; fresh checkouts must reproduce them.
- A review workspace at `/review` with 24 fictional rewrite drafts and 60 fictional JD-label drafts. Approve/reject, edit targets/labels, and download/restore decisions. All 84 prepared drafts were individually AI-reviewed and approved at the owner’s request; they remain explicitly distinct from human review.
- Cross-occupation semantic requirement evidence: closest exact resume quotes with negation/proficiency flags, using a local pinned encoder.
- Career matches from actual resume skills against 3,007 sourced occupation profiles; optional current and target role confirmation; requirements to verify/develop.
- India-first salary comparison with sourced 2025 role/industry references, monthly/LPA/lakh/crore normalization, base-versus-CTC checks, and permitted local benchmark import. See [SALARY_REFERENCES.md](SALARY_REFERENCES.md).
- Trained experimental T5 rewriting, BERT wording analysis with sentence-removal evidence, fitted Snorkel aggregation, and an explicit-role historical GraphSAGE explorer. These are research controls, not validated hiring models.
- Responsive desktop/mobile web screens and a standalone web manifest. A native mobile application is not delivered.

Coverage, similarity, local suggestions, and model attention are not hiring probabilities,
employer ATS reproductions, fairness certifications, or causal explanations.
[STAGE_STATUS.md](STAGE_STATUS.md) tracks the remaining original promises.

## Run locally

Node.js 22+ and Python are required. This checkout is configured and tested with Node 22,
Python 3.13, and an RTX 4050 laptop GPU with 6 GB VRAM and about 24 GB system RAM.

```powershell
npm ci
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ocr.txt
# Install/update Ollama, then explicitly download the local model:
ollama pull gemma4:e4b
# Only create .env if you do not already have one:
Copy-Item .env.example .env
npm run dev:all
```

Open http://localhost:3000. Native frontend and backend bind to loopback.
The npm backend/test scripts select this project's `.venv` automatically.
No provider API key is required. Model and package downloads require internet;
resume processing stays on this machine. Public job search uses internet separately.

The default is Gemma 4 E4B. On this laptop it completed a three-perspective fictional
review in 34.89 seconds and proposal generation/validation in 7.42 seconds. Larger
inputs and cold loads can take longer. These measurements do not establish task quality.
Ollama can use system RAM/CPU offload; a working run does not mean the model fits entirely
in the 6 GB GPU memory. No API key is required.

| Variable | Purpose |
| --- | --- |
| OLLAMA_BASE_URL | Native default http://127.0.0.1:11434; Compose sets http://ollama:11434. Only loopback, Docker Desktop's host gateway, and this exact Compose service address are accepted. |
| OLLAMA_MODEL | Downloaded local model tag, default gemma4:e4b. Cloud-backed tags/aliases are rejected. |
| OLLAMA_CONTEXT | Local context window, 8192 by default, bounded to 8192–16384. |
| OLLAMA_TIMEOUT | Per-generation timeout in seconds, default 300, bounded to 15–300. |
| REVIEW_SIGNING_KEY | Optional shared 32+ character signing secret for multiple backend workers. Otherwise proposals expire on process restart. |
| BACKEND_URL | Next.js proxy target, default http://127.0.0.1:8001. Set before build/dev. |
| BACKEND_PORT | Backend port, default 8001. |
| GAT_MODEL_DIR | Evaluated local semantic artifact directory, default ml/models/gat_entities. |
| API_AUTH_TOKEN | Optional private access token, 32+ characters, entered in the UI and kept in page memory. |
| CORS_ORIGINS | Allowed direct browser origins; does not replace authentication. |

Local generation accepts at most 14 KB of serialized prompt data; longer documents
still receive baseline analysis. The transport never silently slices the resume.
Only one generation runs per backend worker. Rewrite generation may use two calls.
Exact approved action-phrase rewrites keep the remaining source characters unchanged.
Free-form edits require model comparison as well as protected-fact checks.
Proposal generation considers up to 20 existing bullets and returns at most six edits.

## Run everything with Docker

With Docker Desktop running in WSL2 mode and NVIDIA GPU support available, run:

```powershell
docker compose up -d
```

Open http://localhost:3000. Compose starts Ollama, prepares `gemma4:e4b`, and then
starts the backend and frontend. No separate Windows Ollama application or native
`npm run dev:all` process is needed for this setup. The first launch builds missing
app images and downloads the model if absent; allow several minutes and internet
access for that initial setup. Later launches reuse the saved model without pulling
it again. The selected model is controlled by `OLLAMA_MODEL` in `.env`.

The image is pinned to Ollama 0.21.1, whose
[release notes](https://github.com/ollama/ollama/releases/tag/v0.21.1) include the
Gemma 4 structured-output fix for `think=false`. Older native Ollama versions can
ignore the requested JSON schema and cause AI review to fall back to baseline.

Ollama uses the NVIDIA GPU with CPU/system-memory offload when necessary. Its model
files live in the `ollama_data` named volume. `docker compose down` preserves that
volume; `docker compose down -v` deletes it and requires another model download.
The Ollama API is available only inside the Compose network, and cloud features
are disabled. The app remains at localhost ports 3000 and 8001. Services restart
with Docker Desktop unless explicitly stopped.

```powershell
docker compose ps --all
docker compose logs -f ollama-init
docker compose exec ollama ollama list
docker compose exec ollama ollama ps
# Exercise the app and local Gemma with fictional inputs:
.\.venv\Scripts\python.exe scripts/docker_smoke.py --with-ai
```

`ollama-init` exiting with code 0 is expected: it checks/prepares the model and
finishes. The three long-running services are `ollama`, `backend`, and `frontend`.
Docker's baseline backend image does not include the optional research dependencies,
checkpoints, or career encoder/catalog; their controls report unavailable until
those assets are installed/mounted. Native startup remains available for the full
locally installed research environment.

## API

- `GET /health`: actual baseline/OCR/local-model/semantic/research capability states.
- `POST /api/analyze`: resume, jobDescription, optional useAI (local Gemma), useSemantic (GAT experiment), useResearch (BERT/Snorkel experiments), useEvidence (semantic passage retrieval). The UI enables passage retrieval by default.
- `POST /api/analyze-pdf`: multipart file, jobDescription, useAI, useSemantic, useResearch, useEvidence, useOCR, pdfPassword.
- `POST /api/export-pdf`: resume text; returns a clean embedded-font, shaped PDF. Unsupported characters/scripts fail visibly.
- `POST /api/rewrite-proposals`: resume, jobDescription, and optional engine (gemma or t5); returns reviewable edits and an expiring signed proposalToken.
- `POST /api/optimize-preview`: source resume/jobDescription, acceptedEdits, template, optional proposalToken. Checks signature, expiry, source/job binding, duplicate/conflicting/stale IDs, and factual invariants.
- `GET /api/jobs?source=arbeitnow|remoteok&query=...&limit=10`: attributed public listings.
- `GET /api/career-roles?query=...` and `/api/career-explore?code=...`: local ESCO role lookup and historical research ranking, requiring an explicit role selection.
- `POST /api/career-recommendations`: resume, optional currentRoleCode and up to three targetRoleCodes; returns sourced skill evidence/development needs.
- `GET /api/salary-references` and `POST /api/salary-compare`: India reference lookup and explicit offer/reference comparison.
- `/api/graph-data/{id}` returns 410; use the graph in the same analysis response.

`analysis.atsScore` means recognized-skill coverage and can be null. GAT results remain
under `detailed_results.semantic_matching`; local foundation-model review remains under
`detailed_results.local_review`. Evidence-linked Gemma career ideas are separate from the historical GraphSAGE experiment.
Coverage sensitivity, learned entity removal, and attention are labeled separately.
Synthetic-label isotonic calibration is fitted on validation only and evaluated on test;
it does not estimate real hiring probability.
Accepted-edit reanalysis runs the deterministic baseline again; local review is not rerun automatically.

## Review the prepared examples

Open http://localhost:3000/review. The examples are AI-authored fictional practice data,
with no real candidates, vacancies, or hiring outcomes. The prepared records start with
explicit AI approvals; edits reset decisions, and your approval records a human attestation.
Download decisions before leaving the page; Restore progress accepts that downloaded file.

After reviewing a task, import the downloaded decisions:

```powershell
.\.venv\Scripts\python.exe -m data.review_queue import-decisions --bundle "C:\path\careerlens-review-decisions.json" --output Datasets/reviewed/session-1
```

A task exports a reviewed CSV and checksum-bound provenance only after every example
has an explicit approve/reject decision and named reviewer attestation. Names are not
independently verified identities. Human approval of this small templated synthetic set
can exercise training; it does not establish real-world quality or deployment readiness.

For separately licensed/consented source data, use `python -m data.review_queue seed
--help`. Optional `--local-drafts` runs Gemma teacher generation; drafts remain unreviewed.
No web upload enters a training queue automatically.

## Reproduce research models

Install a matching PyTorch wheel, then `requirements-ml.txt` (includes pinned English
spaCy NER). This machine uses CPU PyTorch so training does not compete with Ollama for
GPU memory. The trained files are installed locally and ignored by Git; final manifests
record their checksums, sources, splits and evaluation. Fresh checkouts must reproduce them.

```powershell
.\.venv\Scripts\python.exe -m data.acquire_public
.\.venv\Scripts\python.exe -m training.train_semantic --epochs 12 --jobs 300 --resumes 3000 --graph-mode entity-v1 --output ml/models/gat_entities
.\.venv\Scripts\python.exe -m data.acquire_jobhop
.\.venv\Scripts\python.exe -m data.acquire_esco
.\.venv\Scripts\python.exe -m data.acquire_career_profiles
.\.venv\Scripts\python.exe -m training.build_career_index
.\.venv\Scripts\python.exe -m training.train_career --people 10000 --epochs 60
# Explicit AI-reviewed synthetic experiment; no human-review claim:
.\.venv\Scripts\python.exe -m training.train_rewriter --data data/review_examples/ai-approved/rewrites-reviewed.csv --provenance data/review_examples/ai-approved/rewrites-reviewed.provenance.json --allow-ai-reviewed-synthetic --epochs 15
.\.venv\Scripts\python.exe -m training.train_bias --data data/review_examples/ai-approved/jobs-reviewed.csv --provenance data/review_examples/ai-approved/jobs-reviewed.provenance.json --allow-ai-reviewed-synthetic
.\.venv\Scripts\python.exe -m training.train_weak_wording
```

For independently human-reviewed data, omit `--allow-ai-reviewed-synthetic` and supply
its reviewed CSV and matching provenance. The default training gate still requires a
human-review attestation. AI-reviewed data can never set `reviewed=true`.

GAT uses licensed synthetic relevance labels. T5/BERT use AI-reviewed fictional data,
with repeated templates across source splits. The historical GraphSAGE ranking trails its frequency
baseline and does not use skills. The operational career matcher instead uses local
semantic retrieval and explicit skill evidence from the public occupation taxonomy. Research models remain `deployment_ready=false`;
explicit experimental inference does not change that status. Real-world calibration,
human factual/writing review, occupation/language robustness, and career fairness require
independent evidence. See [DATA_SOURCES.md](DATA_SOURCES.md) and [STAGE_STATUS.md](STAGE_STATUS.md).

## Privacy and deployment

Candidate content is not intentionally saved or logged by this app. Framework uploads
may spool temporarily and are closed after processing. Parser-worker pipes contain text
and passwords in memory. Browser results last until refresh; downloads are user-controlled.
Local Ollama may retain model context in RAM while loaded (keep-alive five minutes);
review Ollama's own logging/configuration for your deployment. Public feed caches and
fictional/reviewed training datasets and the downloaded ESCO occupation catalog are separate from candidate history.

API bodies are bounded to 6 MB before parsing, with four requests per worker and a
30-second upload deadline. Local model generation is serialized per worker. Provider
errors fall back to visible baseline results without cloud requests. These are local
controls, not distributed rate limiting or a complete hostile-document sandbox.

For a public service, HTTPS, per-user identity/access control, distributed limits,
private networking, log policies, and worker isolation still need deployment work.
Docker Compose manages the frontend, backend, and local Ollama service together;
see the Docker startup section above. Native `npm run dev:all` remains available
with an independently running local Ollama server.
The mobile application scope is documented in [MOBILE_SCOPE.md](MOBILE_SCOPE.md).

## Verification

```powershell
npm test
npm run lint
npx tsc --noEmit
npm run test:pdf
npm run build
.\.venv\Scripts\python.exe -m pip check
# Explicit live test with fictional inputs only:
.\.venv\Scripts\python.exe scripts/local_smoke.py
.\.venv\Scripts\python.exe -m training.evaluate_operational
```

Regression tests mock local generation and make no cloud AI calls. The explicit smoke
script calls local Ollama. Optional research/OCR checks run when their dependencies exist.
Both bundled fonts use SIL OFL; licenses are under `public/fonts/`. The current PDF download is generated locally with ReportLab/HarfBuzz. The legacy
jsPDF helper remains regression-tested but is no longer used by the download button.
Unsupported glyphs and right-to-left scripts are explicitly rejected. Other scripts,
complex layouts, and OCR accuracy need broader benchmarks.
