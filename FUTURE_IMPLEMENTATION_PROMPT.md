# CareerLens: implement the remaining product promises

Progress note: This original brief is preserved as the requested scope. Consult
STAGE_STATUS.md and DATA_SOURCES.md before continuing; parts have since been delivered
or experimentally trained, while reviewed-data/deployment gates remain incomplete.

Continue work on this CareerLens repository. First read README.md, PROJECT_STRUCTURE.md,
and the active code. Preserve the repaired text/PDF workflow, canonical skill matching,
request-specific graphs, visible failures, privacy controls, and clean PDF export.

The current product provides deterministic recognized-skill coverage, entity co-occurrence
graphs, phrase screening, formatting-only resume cleanup, and optional opt-in Groq
feedback from three reviewer perspectives. Research modules are not active production
models. Do not label heuristics, placeholders, or installed packages as trained AI.

Implement the following promises that remain incomplete, in incremental verified stages:

1. Real semantic resume–job matching: connect the NER → graph → encoder → matching
   pipeline to the API. Train/load versioned GAT checkpoints from reviewed datasets;
   handle aliases, context, negation, proficiency, required versus preferred skills,
   experience, education, and unsupported occupations. Compare learned matching with
   the deterministic baseline on held-out resumes and jobs. Report appropriate
   classification and per-query ranking metrics, calibration, limitations, and model
   provenance. Do not equate similarity with hiring acceptance probability.

2. Evidence-based hiring simulation: extend the existing optional reviewer feedback
   into distinct ATS, technical-lead, and recruiter assessments with source quotations,
   strengths, concerns, and actionable recommendations. Check parsing/structure and
   actual qualifications rather than just keyword presence. Define and evaluate any
   stage outcomes against an explicit rubric; do not claim to reproduce an employer's
   private ATS or infer personality, cultural fit, or protected traits.

3. Real ethical resume optimization: rewrite/reorder relevant sections and bullets
   while preserving identity, dates, employers, roles, technologies, metrics, and
   achievements. Separate missing skills from existing skills. Provide proposed edits,
   source evidence, an original/proposed diff, and user acceptance/rejection. Add
   semantic factual validation and adversarial tests; the current token validator is
   only appropriate for formatting. Reanalyze accepted changes and display only
   measured improvements, allowing zero improvement. Keep analysis notes out of the
   exported resume.

4. A trained T5 rewriter and teacher/student distillation: collect licensed or
   consented, human-reviewed source/target pairs without invented metrics. Complete
   dataset preparation, split by source to prevent leakage, fine-tune, evaluate factual
   preservation and writing quality, and load the resulting model for inference.
   Treat synthetic negatives and weak labels as approximations, not verified hiring
   outcomes. Never save or advertise an untrained checkpoint as a completed model.

5. Learned job-description quality/bias analysis: train and evaluate the BERT
   classifier and a real weak-supervision label aggregation pipeline. Support
   experience inflation, vague requirements, exclusionary wording, and salary/role
   mismatch only where appropriate data supports them. Return phrase evidence,
   explanations, suggestions, and uncertainty. Avoid arbitrary quality scores and
   blanket declarations of fairness.

6. Personalized career trajectory: collect appropriate career-transition data,
   train/evaluate the GraphSAGE link predictor, and connect it to the candidate's
   actual role and demonstrated skills. Explain suggested next roles and skill
   development. Show timelines/confidence only when supported by validation. Handle
   cold-start candidates and non-software careers without fixed software-role results.

7. Faithful graph explanations: expose actual model attention and/or tested
   attribution methods with named entities, meaningful relationships, and provenance.
   Distinguish observed co-occurrence from learned relations and attention from causal
   explanation. Make displayed graphs and every explanation correspond to the same
   analysis. Do not generate random importance weights.

8. Complete document handling: retain the working text-PDF upload and export, then
   add scanned-PDF OCR, robust reading order for multicolumn resumes, encrypted/invalid
   document handling, and font coverage appropriate to supported languages. Verify
   exported PDFs visually and by text extraction. ATS compatibility should mean
   concrete, tested parsing/structure checks, not a promised acceptance rate.

9. Original roadmap commitments: add role-specific optimization templates and
   job-board integrations using supported APIs or authorized data sources; integrate
   the existing data-collection utilities into a reproducible reviewed pipeline.
   Scope a mobile application only after the web workflow is dependable. Keep roadmap
   features clearly marked as planned until delivered.

Cross-cutting requirements: consistent API schemas and capability reporting, opt-in
external AI processing, explicit provider/model configuration, bounded requests and
timeouts, documented retention, secure deployment, automated regression checks, and
reproducible training environments. Update documentation to match verified behavior.
Missing credentials or datasets must produce an explicit unavailable state, never
fabricated results. Do not commit secrets or publish/deploy automatically.

Deliver each stage with implementation, relevant tests, measured evaluation where
applicable, and an honest account of remaining limitations. Start with factual rewriting
and evidence-linked assessments, then complete the learned-model pipeline.

## Separate optional extensions (not previously implemented promises)

DOCX import/export, user-controlled saved versions and history, multi-job comparison,
recruiter bulk screening/shortlists, collaboration, and personalized learning resources
are additional product ideas. Do not present them as existing features or expand into
them before completing the stages above unless explicitly requested.
