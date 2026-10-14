# CareerLens

Resume/job evidence review with a minimal black-and-white interface. Baseline analysis
works without API keys, model downloads, or saved candidate history.

## Delivered web workflow

- Text and PDF analysis: 5 MB/file, 30 pages, 50,000 characters/input.
- Canonical skill coverage, aliases, source evidence, and basic negation handling.
  Unsupported job vocabulary returns N/A. Mentioning a skill never proves proficiency.
- Required/preferred qualification checklist, with experience and degree uncertainty.
- Distinct ATS/structure, technical-evidence, and recruiter-clarity rubrics with quotes.
  Optional Groq reviewers must return quotations found in the supplied text.
- Reviewable constrained bullet edits: original/proposed comparison, individual
  acceptance/rejection, source-preserving section templates, and reanalysis. Zero
  improvement is a valid result. No missing skills, numbers, or achievements are added.
- Clean PDF export with embedded Noto Sans, wrapping, pagination, and visible errors
  for unsupported characters. Latin accents, Greek, and Cyrillic are regression-tested.
- Text extraction with a two-column heuristic, password handling, and optional local
  OCR for up to five scanned pages. Review OCR/column order before accepting edits.
- User-triggered job-description import from Arbeitnow, with source attribution,
  HTTPS source links, a six-hour public-feed cache, and no resume sharing/applications.
- Optional locally trained semantic GAT experiment, separate from skill coverage.
  Actual model attention is attached to exact input text chunks, not to unrelated
  entity-graph nodes. Missing weights/dependencies or unsupported job titles return
  explicit unavailable results.

Neither coverage nor similarity is a hiring probability, employer ATS reproduction,
fairness certification, or prediction of individual potential.

[STAGE_STATUS.md](STAGE_STATUS.md) maps each original promise to delivered work and
remaining gates. [DATA_SOURCES.md](DATA_SOURCES.md) records public sources, licenses,
revisions, attribution, rejected sources, and measured experiments. The original
implementation brief remains in [FUTURE_IMPLEMENTATION_PROMPT.md](FUTURE_IMPLEMENTATION_PROMPT.md).

## Setup

Use Node.js 22+ and a separate Python environment. Optional ML/OCR package availability
depends on Python/platform; the recorded experiment used Windows/Python 3.14.5 on CPU.

```sh
npm ci
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
npm run dev:all
```

Open http://localhost:3000; backend defaults to http://127.0.0.1:8001.
Separate commands: `npm run backend`, `npm run dev`.
Optional local OCR: `python -m pip install -r requirements-ocr.txt`.
Docker's default image includes baseline analysis and local OCR, not experimental ML
dependencies/weights. Set INSTALL_OCR=false before building for a smaller baseline image.

```sh
docker compose up --build -d --wait
docker compose ps
# End-to-end proxy checks without external AI calls:
python scripts/docker_smoke.py
# Stop without deleting images:
docker compose stop
# Remove this app's containers/network when no longer needed:
docker compose down
```

The frontend is available at http://localhost:3000 and backend at
http://localhost:8001. Both containers run as non-root users with health checks;
published ports bind only to loopback. Compose reads optional provider/access
configuration from your local .env; that file is never copied into either image.
The frontend uses Next.js standalone output. Python downloads use a persistent
BuildKit package cache and resumable retries. Optional pre-downloaded Linux/Python
3.11 wheels under ignored docker/wheels are used locally; an empty wheels directory
still builds by downloading from the package index. Do not place Windows wheels there.

Copy `.env.example` to `.env` when configuring optional features.

| Variable | Meaning |
| --- | --- |
| BACKEND_URL | Next.js proxy target; default http://127.0.0.1:8001. Set before build/dev. |
| BACKEND_PORT / BACKEND_HOST | Port/host for python main.py; default 8001 / 127.0.0.1. Uvicorn CLI uses its own flags. |
| GROQ_API_KEY / GROQ_MODEL | Server-only optional reviewer credentials/model. Default model: openai/gpt-oss-20b. |
| GAT_MODEL_DIR | Local experiment manifest/checkpoint directory. Default ml/models/gat_experimental. |
| CORS_ORIGINS | Allowed direct browser origins; not authentication. |
| API_AUTH_TOKEN | Optional private-deployment bearer token, at least 32 characters. Enter in the UI access field; never stored in browser storage. |

No API key is required for baseline analysis. Selecting external AI explicitly sends
resume and JD text to Groq; provider retention is governed by its policies. Local
semantic inference and OCR do not upload input or download models during requests.

## API and safety

- GET /health: baseline, OCR, semantic-artifact and research-model status. Configuration
  does not prove provider reachability or model suitability.
- POST /api/analyze: JSON resume, jobDescription; optional useAI and useSemantic.
- POST /api/analyze-pdf: multipart file, jobDescription; optional useAI, useSemantic,
  useOCR, pdfPassword. Passwords are used in memory, not forwarded to providers.
- POST /api/optimize-preview: original resume, jobDescription, acceptedEdits (server-issued
  proposal IDs), template. Rejects stale/unknown/duplicate IDs. Only accepted changes apply.
- GET /api/jobs?query=...&limit=10: source-attributed public listings.
- Graphs come from detailed_results.graph_analysis in the same response. The old
  shared /api/graph-data/{id} endpoint returns 410.

The compatibility analysis.atsScore field means recognized-skill coverage and may
be null. Learned similarity remains at detailed_results.semantic_matching.
Rewrite comparison measures coverage before/after; it does not invent a writing-quality score.

API bodies are capped at 6 MB before multipart/JSON parsing. Four API requests per
worker may run concurrently, excess returns 429; uploads time out after 30 seconds.
Provider review has a 20-second timeout and no automatic retries. These limits are
not a distributed rate limiter or a hard sandbox for hostile PDF processing.

Candidate content is not intentionally saved or logged. Framework PDF upload files
may be temporarily spooled and are closed after handling; browser results remain in
page memory until refreshed. Downloaded resumes are user-controlled files. Public
training datasets/checkpoints and public job-feed caches are separate from candidate history.

For nonlocal deployment: require HTTPS, authentication, edge rate limits, worker
CPU/memory/time limits, private backend networking, and log/body-redaction policies.
The shared bearer option is for private access, not a multi-user identity system.
Docker ports bind to loopback by default. Do not expose an unauthenticated backend.
No deployment or publication happens automatically.

## Reproduce the learned experiments

Install an appropriate torch wheel and `requirements-ml.txt` in a separate environment.
Exact tested package versions and checkpoint SHA256s are recorded in the manifests.
Weights and datasets are ignored by Git; the tracked manifest alone never enables inference.

```sh
python -m data.acquire_public
python -m training.train_semantic --epochs 12 --jobs 300 --resumes 3000
python -m training.train_semantic --evaluate-only --epochs 12 --jobs 300 --resumes 3000

python -m data.acquire_jobhop
python -m training.train_career --people 10000 --epochs 60
```

Acquisition is explicit, revision-pinned, bounded, and preserves source cards and
checksums. Semantic evaluation holds out both resumes and jobs; career evaluation
holds out people and uses training-only role features/edges. Neither experiment
validates hiring outcomes. Career ranking trails the frequency baseline, and source
fairness/cross-market checks are missing, so career suggestions remain disabled.

The semantic encoder uses observed text chunks (at most 32, excluding contact lines
containing @), MiniLM embeddings, and adjacency edges. It is **not** the completed
learned NER/entity-relation pipeline promised in the roadmap. Inference is limited to
explicit job-title families in the manifest; new occupations need reviewed validation.

## Reviewed-data training gates

No suitable reviewed rewriting/bias corpus was accepted during sourcing, so T5 and
BERT capabilities remain unavailable. The prepared training commands are not claims
of trained models or verified quality:

```sh
python -m training.train_rewriter --data reviewed_rewrites.csv --provenance rewrite_source.json
python -m training.train_bias --data reviewed_jds.csv --provenance jd_source.json
python -m data.label_aggregation --votes licensed_votes.csv --provenance votes_source.json
```

Review manifests require source_url, license, task, reviewer, reviewed=true,
label_method, and data_sha256 matching the CSV. Reviewer fields are documented
attestations, not independently verified identities.

T5 CSV: source_id (original resume ID), original, improved, jd_context. At least ten
source resumes; split by source and reject duplicate original bullets across splits.
Fine-tuning selects validation loss and separately evaluates test loss and numeric,
skill, and negation invariants. Those invariants do not prove semantic fidelity or
writing quality, so generated outputs remain research-only and never auto-apply.

BERT CSV: source_id, text, plus reviewed binary label columns named in manifest.labels.
Only supported classes are trained; unsupported categories remain unavailable.
At least 50 source documents, disjoint holdouts, validation thresholds, per-label
AUC/F1/precision/recall/Brier. Phrase attribution and context/fairness evaluation
still gate deployment. No overall quality/fairness score is fabricated.

Snorkel CSV: source_id,text,lf_* votes (-1 abstain/0/1), at least three documented
labeling functions. Provenance names source/license/task, label_definitions, data_sha256.
It fits a real LabelModel and leaves all-abstain rows unlabeled; weak posteriors need
independent human evaluation. Teacher-generation utilities remain opt-in research,
require licensed inputs and human review, and are not run by the web app.

## Verification

```sh
npm test
npm run lint
npx tsc --noEmit
npm run test:pdf
npm run build
npm audit
docker compose config --quiet
```

No live AI-provider requests are made by tests. ML/OCR tests skip when dependencies
are absent. PDF export is verified by full text extraction and rendered-page review;
test artifacts live under ignored tmp/pdfs. Browser interaction must also be manually
checked when browser access is available.

Noto Sans is bundled under SIL OFL; see public/fonts/OFL.txt. Scripts such as Devanagari
and emoji are not silently approximated: additional fonts/shaping tests are still needed.
