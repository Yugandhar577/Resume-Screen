# Product promises: verified stage status

This is an incremental delivery, not completion of every original promise.
The minimal black-and-white UI and repaired baseline remain intact.

| Original promise | Delivered in this stage | Still needed |
| --- | --- | --- |
| Semantic matching | Trained local MiniLM/text-graph/GAT experiment, opt-in API/UI, checksum/provenance, disjoint heldouts and baseline comparison; evidence rules for negation/required/preferred/proficiency uncertainty | Reviewed real relevance labels, calibrated estimates, complete learned NER/entity-relation mapping, multilingual/occupation generalization and robustness |
| Hiring simulation | Three distinct quoted rubrics, qualifications checklist; optional provider quotations verified against input | Human evaluation of rubric accuracy; no employer ATS or hiring-stage outcome is claimed |
| Ethical optimization | Constrained introductory bullet rewrites, side-by-side diff, accept/reject, intact-section templates, actual reanalysis including zero gain | Validated free-form rewriting and adversarial semantic factual evaluation beyond the supported transformations |
| T5/distillation | Review/license/checksum gates, source-disjoint training/validation/test preparation, local artifact loading contract, no untrained inference or fabricated improvement scores | Accepted reviewed pairs, actual fine-tuning run, human factual/writing-quality benchmarks and safe web integration |
| Learned JD analysis | Real fitted Snorkel aggregation, preserved abstentions; review-gated BERT training/evaluation path; removed legacy fake probabilities/quality score | Suitable reviewed licensed labels, actual BERT run, phrase attribution, regional compensation evidence for salary checks and deployment validation |
| Career trajectory | Trained GraphSAGE research benchmark on pinned JobHop, person-disjoint evaluation and frequency baseline | Better ranking, candidate-role/skill mapping, fairness and cross-market validation. No personalized output/timeline/confidence is enabled |
| Graph explanations | Input-specific observed entity co-occurrence graphs, separate actual GAT text-chunk attention with exact quotes | Validated causal/perturbation attribution and learned entity relationships; attention is not causal explanation |
| Document handling | Local OCR, inferred two-column order, correct-password handling, bounded pages/size, clean paginated export and tested Latin/Greek/Cyrillic font coverage | Complex layout and OCR accuracy benchmark; Indic/other scripts and shaping coverage; hardened hostile-document worker sandbox |
| Roadmap/templates/job boards | Engineering/data/general section templates and supported Arbeitnow import, pinned reproducible public acquisition utilities | Wider authorized sources and human-review pipeline. Mobile remains planned after dependable web UX validation |

Cross-cutting: pre-parser body/upload bounds, per-worker concurrency limits, optional
bearer access, loopback defaults, provider timeouts, no-store results, no intentional
candidate persistence/logging, truthful capability reporting, automated regression
checks. Production still needs TLS, distributed rate limits, identity management,
worker sandbox/resource limits and operational policy review.

## Verification performed

- 42 passing API/engine/research/OCR/proposal and Snorkel regression checks; see tests/.
- Local trained GAT API inference with actual source-text attention; no live provider calls.
- ESLint, TypeScript and Next.js production build.
- PDF export text extraction: all 150 numbered entries, final line, Latin accents,
  Greek and Cyrillic preserved; all four rendered pages visually inspected.
- npm audit reported zero vulnerabilities. Docker frontend/backend images were built
  and started locally, with both health checks passing. Proxy checks passed for text
  analysis, accepted edits, PDF text upload, font delivery and scanned-PDF OCR.
  Containers include local OCR; experimental ML dependencies/weights remain optional.
- Live public Arbeitnow API returned source-attributed listings; no candidate content was sent.
- Interactive browser QA was unavailable because browser access was denied. No alternate
  browser access was used; manual UI acceptance still remains.

Training results and source-selection constraints are in DATA_SOURCES.md. Checkpoints
are local ignored artifacts; checked-in provenance does not substitute for the weights.

## Next concrete gate

Obtain licensed/consented source documents and have qualified reviewers create or
verify rewriting targets and JD labels. Public synthetic inputs can seed a review
queue, but machine-generated targets must not be relabeled as human-reviewed. Train
only after the checksum-bound review manifest is complete; publish neither candidate
data nor consequential recommendations automatically.

Separate optional extensions (DOCX, history, bulk screening, collaboration, learning
resources) were not added. The original implementation brief remains unchanged.
