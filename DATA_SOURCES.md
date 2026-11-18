# Public sourcing and measured experiments

Sources were searched online at the user's request. A stated license is recorded,
not treated as evidence of label accuracy or real hiring validity. Downloads are
explicit; candidate uploads are never used as training data automatically.

## Accepted: synthetic matching experiment

[Michael Ozon's candidate-matching-synthetic](https://huggingface.co/datasets/michaelozon/candidate-matching-synthetic)
declares MIT licensing: 10,000 synthetic resumes, 2,500 jobs and synthetic relevance
lists generated from skill rules. Source author: Michael Ozon. It is **not** reviewed
hiring outcome data. Source card is preserved with the downloaded files.

Pinned revision: `178ab864dcad9910c5670d43e4bdbbb901a11f18`.
`python -m data.acquire_public` records source URLs, byte sizes and SHA256s.

Encoder: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
Apache-2.0, pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
The original text-chunk GAT benchmark is retained below for comparison. The active
entity-v1 model uses pinned pretrained spaCy en_core_web_sm 3.8.0 (MIT), canonical
career spans, source containment/adjacency relations, frozen MiniLM embeddings, and
a separately trained two-layer Siamese GAT. The NER is general English, not a
career-fine-tuned model; relationships are observed, not learned factual assertions.

Run: seed 42, CPU, 12 epochs, sampled 3,000 resumes and 300 jobs. Both resume and job
IDs are disjoint across 60/20/20 splits. Negatives are sampled non-listed synthetic
pairs, not real rejections. Test: 438 pairs, 60 job queries, small sampled candidate pools.
Both classification thresholds are selected using validation F1 only. Ranking ties
use label-independent SHA256 ordering, never insertion order or labels.

| Held-out metric | Learned GAT | Canonical skill baseline |
| --- | ---: | ---: |
| AUC | 0.9314 | 0.6643 |
| F1 | 0.7660 | 0.4906 |
| Accuracy | 0.8744 | 0.7534 |
| Recall@5 | 0.8003 | 0.7039 |
| MRR | 0.7111 | 0.5867 |
| Brier diagnostic | 0.1311 | 0.2202 |

Cosine is mapped to [0,1] only for the Brier diagnostic; neither method is calibrated
as hiring probability. Results do not generalize automatically to real candidates,
full ranking pools, unsupported occupations, or other languages. The source's 24
job-title families are recorded in the manifest; unsupported titles explicitly decline
semantic inference. Experiments are opt-in and never replace the baseline coverage.

Legacy text experiment: `ml/models/gat_experimental/manifest.json`, ignored `model.pt`, local
`Datasets/models/minilm`. Reproduce with the commands in README.md. Package versions,
checksum, source attribution and evaluation are preserved in the manifest.

## Accepted: career-transition research only

[JobHop](https://huggingface.co/datasets/aida-ugent/JobHop), CC BY 4.0.
Attribution: Iman Johary, Guillaume Bied, Alexandru C. Mara, Tijl De Bie, Ghent
University; source data provided by VDAB, the Flemish Public Employment Service.
JobHop v2 paper: [A Large-Scale Career Trajectory Dataset from Unstructured Resumes](https://arxiv.org/abs/2607.11715).

Pinned revision: `766921f12b66eb886841393ab0400321e0373cb4`. Downloaded only the
structured v2 train/validation/test parquet files and source card; acquisition records
checksums/attribution. No attempts at re-identification, linkage, or protected-attribute
inference were made. Quarter dates are used only to order non-overlapping transitions.

The authors describe pseudonymized VDAB careers, LLM extraction and imperfect ESCO
assignment, population/geographic limitations and historical structural bias. They
call for fairness evaluation before deployment, and warn against sole/decisive use
for high-stakes individual decisions. The public records lack candidate skill evidence.

Run: seed 42, CPU, 60 GraphSAGE epochs; deterministic person subsamples of 10,000
training and 2,000 each validation/test. Source splits are person-disjoint. Training
alone constructs role vocabulary, code-prefix features and graph edges. Recurrent
role links across people are allowed: this is not an unseen-edge benchmark. Unknown
or infrequent-role transitions are excluded and counted. Uniform negative sampling
does not establish that a career move is impossible.

Test: 4,281 retained positive transitions, 8,562 balanced classification pairs,
1,271 candidate occupation codes; 570 extracted transitions excluded for vocabulary
coverage. Validation alone selects the classifier threshold 0.39.

| Held-out metric | GraphSAGE | Transition-frequency baseline |
| --- | ---: | ---: |
| AUC | 0.8409 | 0.6976 |
| Recall@5 | 0.1196 | 0.1444 |
| MRR | 0.0791 | 0.1013 |

GraphSAGE F1 0.7602, average precision 0.8609, uncalibrated Brier 0.1606. Its ranking
is worse than the frequency baseline. Checkpoint and evaluation are real, but
`deployment_ready=false`: the app exposes an explicitly selected-role historical
research ranking, with its poor baseline comparison visible. It does not produce
individual suitability judgments, timelines or calibrated confidence. Evidence-linked
Gemma career ideas are a separate local foundation-model workflow. See `ml/models/career/manifest.json`.

## Not accepted as reviewed training data

- [JABD](https://github.com/marcelomendoza/jabd): human-annotated job-ad wording, but
  its [upstream OpenRAIL-D license](https://raw.githubusercontent.com/marcelomendoza/jabd/main/LICENSE)
  restricts consequential uses involving identified individuals. It was not downloaded
  or incorporated into this candidate-screening workflow. A downstream model card's
  simpler license declaration would not resolve upstream restrictions.
- [vsr9awc/resume-optimizer](https://huggingface.co/vsr9awc/resume-optimizer): synthetic
  teacher-generated rewriting and uncertain upstream training-source permissions do
  not supply the requested licensed human-reviewed factual T5 pairs. Not downloaded.
- Public Kaggle/resume scraping utilities without clear source permissions, consent
  and label review were not treated as training authorization.

No suitable reviewed rewriting/bias corpus was accepted in this search. This is not
a claim that none exists. Default training paths require checksum-bound human-review/source attestations.
The owner subsequently authorized AI review of project-authored fictional examples;
that explicit experimental path is documented below and does not create human labels. Salary mismatch additionally needs regional, occupation-specific compensation
evidence; technology count is not compensation evidence.

## Public job-description import

[Arbeitnow's supported API](https://www.arbeitnow.com/blog/job-board-api) is used
for user-triggered JD import, not training, scraping or job applications. It requires
no API key and primarily serves European listings. Source backlink/attribution is
preserved per its [terms](https://www.arbeitnow.com/terms). A fixed endpoint, 15-second
timeout, 4 MB fetch bound, HTTPS link allowlist, script/style removal and six-hour
cache bound the integration. No resume or candidate text is sent to the job board.

## Local Gemma and review examples — 14 October 2026

The app now uses the explicitly downloaded `gemma4:e4b` through local Ollama,
with cloud models/aliases disabled. The tag and digest are reported in local review
responses. Model source: https://ollama.com/library/gemma4; license information is in
that model listing. This foundation-model use is separate from task-specific T5/BERT
training. Both trained research checkpoints now exist locally, with their explicit
AI-review provenance and deployment limits.

`data/review_examples.json` and `data/review_examples/` contain project-authored fictional
practice examples: 24 draft rewrites from 12 source IDs and 60 draft JD labels. Their
source/usage metadata explicitly identify synthetic origin and AI review. All 84
examples were read and approved as AI-reviewed, at the owner’s request. The browser
allows a person to inspect, correct and replace those decisions. No human-review
attestation was fabricated.
No candidate uploads were included. Templates can recur across source IDs, so these
examples are suitable for review/training-pipeline exercises rather than real-world
quality estimates. Human approval does not erase that limitation.

The semantic GAT experiment was reproduced on this machine with the pinned public source
and sentence encoder, 12 epochs, 300 jobs, and 3,000 resumes. The reported held-out
metrics matched the table above; current environment/checksum are in the model manifest.
Weights and encoder are local ignored files, not shipped in Git.

Remote OK was added using its supported public API: https://remoteok.com/api.
Official feed/attribution guidance: https://remoteok.com/faq; terms: https://remoteok.com/legal.
Source names and original posting links are displayed, queries filter the cached public
feed locally, no resume is sent, and no listings are incorporated into training.
The implementation uses a fixed HTTPS endpoint, a 15-second fetch timeout, a 4 MB bound,
source-host URL checks, sanitized descriptions, and a six-hour per-worker cache.

The career benchmark was also reproduced on 14 October 2026. Current test AUC is 0.8409002, F1 is 0.7609745, Recall@5 is 0.1195982, and MRR is 0.0793154. Its ranking still trails the frequency baseline, so deployment remains disabled. Small numerical differences from the earlier table are preserved in the updated manifest rather than treated as a quality improvement.


## Added trained experiments and taxonomy — 14 October 2026

Active entity graph GAT (`ml/models/gat_entities/manifest.json`), same licensed
3,000-resume/300-job sample and disjoint splits as above, 12 epochs:

| Held-out metric | Entity/source GAT | Canonical baseline |
| --- | ---: | ---: |
| AUC | 0.9297 | 0.6643 |
| F1 | 0.7368 | 0.4906 |
| Recall@5 | 0.8078 | 0.7039 |
| MRR | 0.7111 | 0.5867 |

Actual entity/source attention is exposed; up to eight resume entity nodes are removed
one at a time and inference rerun. Source chunks remain present during removal.
Attention and perturbations measure model behavior, not causation. Graphs encode at
most 16 source chunks plus 16 entities; names, exact source quotes, offsets, extraction
method and observed relation types are returned. Supported title families are explicit.
Validation-only isotonic calibration evaluates synthetic benchmark labels/prevalence;
its reliability bins, Brier score and calibration error are stored in the manifest.
The observed test Brier is 0.08921 and calibration error is 0.02591 on these
synthetic labels. These are not real-world hiring probabilities.

T5-small: pinned google-t5/t5-small revision
`df1b051c49625cf57a3d0d8d3863ed4d13564fe4`, 15 epochs, 24 AI-reviewed fictional pairs
from 12 sources. Split: 16 train / 4 validation / 4 test rows, source-disjoint.
Test loss 0.01785; measured number/recognized-skill/negation preservation each 4/4.
An earlier 3-epoch pass preserved numbers in 3/4; the longer run used validation loss
for checkpoint selection. The same four test examples were inspected after the
initial run and reused after extending training, so final results are not an untouched
model-selection holdout. This tiny repeated-template evaluation is not independent
writing-quality or factual-equivalence validation. Generated candidates still pass
strict protected-fact checks, Gemma semantic comparison, signing and user acceptance.

BERT-base-uncased: pinned revision `86b5e0934494bd15c9632b12f734a8a67f723594`,
3 epochs, 60 AI-reviewed fictional source descriptions; source-disjoint 36/12/12.
Thresholds selected on validation only. Held-out category metrics:

| Category | AUC | F1 | Brier |
| --- | ---: | ---: | ---: |
| Vague | 1.0000 | 1.0000 | 0.1416 |
| Potentially exclusionary wording | 0.7143 | 0.7692 | 0.2288 |
| Experience inflated | 1.0000 | 0.9231 | 0.1680 |

Raw activations are not calibrated fairness probabilities. Repeated wording templates
cross source splits, so these results do not establish contextual generalization.
The API separately reports actual sentence-removal sensitivity (up to 12 sentences),
not proof that a highlighted phrase caused a real-world decision.

Snorkel LabelModels: fitted on the same 36 training documents, three documented,
correlated rule votes per category; evaluated on disjoint 12/12 source documents.
Rules were authored after inspecting the examples; this is not a blind generalization
test. All-abstain inputs stay unlabeled. Safe numeric serialization reproduces Snorkel
inference, without pickle. Perfect classification on repeated templates is expected
and is not evidence of general fairness assessment. Patterns, split counts, numeric
checkpoint hash and limitations are recorded in `ml/models/weak_labels/manifest.json`.
Weak posteriors never become reviewed labels or overwrite the BERT/source data.

ESCO: [European Commission web-service API](https://esco.ec.europa.eu/en/use-esco/use-esco-services-api),
explicit `v1.1.2` to match JobHop’s taxonomy. Downloaded 3,007 English occupational
names/codes/URIs, with attribution and snapshot checksums. The local catalog supports
explicit user-selected role mapping and no candidate text is sent to this source.
Classification reuse notice: [European Commission legal notice](https://commission.europa.eu/legal-notice_en).
Taxonomy API software is not redistributed. Acquire using `python -m data.acquire_esco`;
the snapshot is ignored by Git. No unlicensed salary data or inferred candidate skill
history is added.


## India-first operational sources

The career matcher now uses 3,007 ESCO v1.1.2 profiles, including essential/optional
skill relations and descriptions, from the official public API. Public pages are
acquired with bounded requests, retry/resume caching and page hashes. English fields
are extracted and indexed locally with the already pinned MiniLM encoder. Attribution:
European Commission ESCO; EU-owned content follows the Commission's CC BY 4.0 reuse
notice. Extraction/reformatting/embedding are project transformations. No candidate
text is sent to this source or stored in the public index. This is occupation/skills
retrieval, not a trained forecast of Indian career outcomes.

India salary numeric references are attributed in SALARY_REFERENCES.md. The bundled
subset has 8 recruiter-reported role/experience CTC ranges and 20 industry/experience
references from 2025 reports. They retain source dates and coverage; industry
compensation basis remains unspecified. No present-day inflation, city adjustment,
representative survey claim or salary percentile is invented. These facts are not
used as model-training labels; complete reports are not redistributed.

Additional matching sources were checked during this pass. cnamuangtoun's source
has no dataset card/license recorded at revision 08978e21714984bb417547d2c0f9b477f5298163;
the downstream 0xnbk score set uses algorithmic, not human relevance scores.
jminc/resume-matching-dataset-v2 explicitly describes synthetic teacher-generated
Korean examples; resume-atlas is a classification source, not reviewed rewriting
pairs or relevance judgments. None was relabeled as independently human-reviewed
candidate/job evidence. Existing research artifacts retain their limitations.
