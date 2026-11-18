# Current scope status — 14 October 2026

The local desktop web implementation and experimental model integration are delivered.
The original research promises still require independent real-world data and evaluation;
AI approval of fictional examples cannot complete those empirical requirements.

| Original scope | Implemented and exercised | Evidence still required |
| --- | --- | --- |
| Semantic matching | Cross-occupation local semantic requirement-to-resume passage retrieval with exact quotes/qualifier flags; local pretrained NER → exact entity/source graph → MiniLM → trained GAT; aliases, contextual baseline requirements, unsupported-title handling; source/job-disjoint evaluation against baseline; synthetic-label calibration | Reviewed real relevance labels, occupation/language robustness and real-market calibration |
| Hiring simulation | Three distinct evidence-linked rubrics and local E4B reviewers; parsing, qualifications, quoted strengths/concerns/recommendations | Independent factual/rubric usefulness assessment; no private ATS or employer outcomes predicted |
| Ethical optimization | Gemma proposals, strict protected facts, exact approved-transformation certificates or semantic comparison, signed acceptance, source diff, section templates, actual reanalysis and clean export | Broader adversarial/human equivalence and writing-quality benchmarks |
| T5/distillation | Licensed fictional examples, recorded AI approvals, explicit synthetic training gate, source-disjoint T5 fine-tuning/evaluation, local trained inference through shared validation/signature path; local teacher utility | Independently human-reviewed realistic examples and writing-quality evaluation; research remains deployment_ready=false |
| Learned JD wording | Trained BERT, category holdout metrics, actual sentence-removal evidence, fitted Snorkel numeric aggregation, explicit abstentions and opt-in UI | Real contextual wording labels and deployment-context robustness/calibration |
| India salary comparison | 8 sourced role CTC references, 20 industry/experience references, explicit base/CTC/units/experience checks, source dates/uncertainty, local custom-benchmark import | Broader current occupation/city coverage needs appropriate sources; no personal salary prediction |
| Career exploration | Sourced skill-based career matching over 3,007 ESCO occupations with optional current/target role confirmation and exact resume evidence; evidence-linked local E4B ideas; trained/evaluated GraphSAGE; explicit local ESCO role search and historical research ranking; cold-start decline | GraphSAGE has no candidate skill input, trails frequency ranking, and lacks fairness/cross-market validation. No suitability/timeline/confidence claims |
| Graph explanations | Request-specific observed graphs, named entity spans, actual learned attention, actual entity-node removal reruns; separate actual coverage-score perturbation | No causal claim or validated learned relation semantics; pretrained English NER is not career-fine-tuned |
| Documents | Text/PDF/password/OCR/column workflows, editable extraction, bounded disposable parser; Windows memory/CPU/process and Linux address-space/CPU limits; shaped embedded-font Unicode PDF export | Broader OCR/layout/script benchmarks and a hostile-document security sandbox |
| Roadmap | Six role templates, Arbeitnow/Remote OK import with attribution, responsive 320 px/desktop UI and web manifest; mobile scope in MOBILE_SCOPE.md | Native phone app and phone/laptop pairing are a future release; public deployment is not authorized |

## Default local AI

Gemma 4 E4B is configured in `.env`, `.env.example`, the local transport and Docker
configuration. Live fictional smoke: three reviewers returned in 34.89 seconds;
rewrite generation/validation/application completed in 7.42 seconds. Actual laptop:
RTX 4050 6 GB VRAM, about 24 GB RAM; working inference may include CPU/RAM offload.
The per-generation timeout is 300 seconds. No API key or cloud fallback is used.

## Draft review and trained models

All 24 rewrites and 60 JD drafts were read and AI-approved at the owner’s explicit
request. The audit is `data/review_examples/AI_REVIEW.md`. Checksummed approved datasets
are under `data/review_examples/ai-approved/`. Their provenance records `review_type=ai`,
`ai_reviewed=true`, `reviewed=false`. Default human-review gates remain intact; synthetic
AI training requires an explicit flag. No candidate upload entered training.

T5, BERT, Snorkel, entity GAT and GraphSAGE artifacts are genuinely trained locally.
Their final manifests record sources, hashes, holdout metrics and limitations; large
weights/datasets are ignored by Git and must be reproduced in a fresh checkout.
T5’s final four-example reused holdout preserved measured numbers, recognized skills and
negation in 4/4, which does not establish semantic equivalence. BERT held-out F1 is
1.0000 / 0.7692 / 0.9231 for the supported categories. GraphSAGE ranking still trails
its frequency baseline. None is advertised as validated for consequential hiring.

## Verification

76 automated tests passed, including AI-vs-human provenance gates, T5 protected-fact
rejection, opt-in research separation, exact entity spans/relations, weak-label
abstention and actual Windows memory-limit enforcement. Production build, lint,
local checkpoint inference and local E4B smoke are exercised. Browser checks exercised E4B + BERT/Snorkel + entity GAT together, explicit ESCO
role selection and historical inference, T5 generation with conservative rejection,
signed acceptance, actual before/after reanalysis, and PDF/text downloads preserving
accepted and unaccepted facts. No overflow was observed at 320 px or 1280 px; no browser
console errors were recorded. Isolated OCR passed under Windows resource limits.
Reloaded BERT weights reproduced the recorded held-out metrics.
The native local setup is the tested path; revised Docker research images are not
claimed as tested or supplied with optional research weights/dependencies.

## Local privacy and deployment

Native servers bind to loopback. Bounded requests, optional private bearer access,
per-worker concurrency, no-store API responses and no intentional candidate history
are implemented. Public hosting still requires actual hosting configuration, HTTPS,
identity management, distributed limits, stronger document isolation and operations.
No app store release or public deployment has been performed.

The preserved original brief is FUTURE_IMPLEMENTATION_PROMPT.md. Optional DOCX,
history, bulk screening and collaboration extensions were not added to this scope.


## India-first completion pass

Operational workflows now include cross-occupation semantic evidence retrieval, exact
transformation certificates for approved rewrites, a 3,007-profile skills-based career
index, current/target occupation selection, and India salary extraction/comparison.
These workflows do not depend on the historical GraphSAGE ranking being valid.

Salary references retain 2025 source dates. Role references describe recruiter-reported
CTC offers; industry references do not specify compensation basis and are displayed
as references without a mismatch conclusion. Monthly/annual, lakh/crore, currency,
base/CTC, experience brackets, conflicting units and multiple components are checked.
No city premium, inflation update, sample size, percentile or confidence is invented.
A permitted local benchmark JSON can extend coverage without a provider API key.

Fresh authored stress evaluation: 96 exact rewrite cases passed; 480 altered fact
cases were rejected; 18 qualification/negation cases passed. These test selected
invariants and evidence rules, not real hiring validity or human writing quality.
Results are in evaluation/operational_stress.json. See SALARY_REFERENCES.md for data
coverage and import format. Candidate data is not used in the downloaded career index.

Independent human/real-world validation of learned research models remains empirical
work. This pass does not relabel AI approvals, synthetic test cases or public taxonomy
relations as human-reviewed candidate outcomes.


Final India-first browser checks exercised source-preserving semantic evidence,
automatic skill-based role retrieval, explicit Data Scientist target selection, and
an INR CTC comparison against a sourced role reference. No horizontal overflow was
observed at 320 px or 1280 px and no browser console errors were recorded. A missing
terminal full stop from an otherwise exact approved generated transformation is
restored from the source; other missing/changed facts are never repaired automatically.

Local benchmark import/restore was also exercised in the browser using an explicitly labeled test fixture; the published reference was restored afterward. Live trained T5 generation passed exact-transform certification and signed acceptance after source punctuation restoration.
