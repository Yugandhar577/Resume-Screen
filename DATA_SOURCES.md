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
Frozen sentence embeddings feed a trained two-layer Siamese GAT over observed text
chunks and adjacency edges. This is not a trained named-entity/relation extractor.

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

Artifacts: `ml/models/gat_experimental/manifest.json`, ignored `model.pt`, local
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
`deployment_ready=false`: no career suggestions, suitability judgments, timelines,
or individual confidence are exposed by the web app. See `ml/models/career/manifest.json`.

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
a claim that none exists. Training paths require checksum-bound human-review/source
attestations. They remain untrained/unavailable until appropriate data and evaluation
exist. Salary mismatch additionally needs regional, occupation-specific compensation
evidence; technology count is not compensation evidence.

## Public job-description import

[Arbeitnow's supported API](https://www.arbeitnow.com/blog/job-board-api) is used
for user-triggered JD import, not training, scraping or job applications. It requires
no API key and primarily serves European listings. Source backlink/attribution is
preserved per its [terms](https://www.arbeitnow.com/terms). A fixed endpoint, 15-second
timeout, 4 MB fetch bound, HTTPS link allowlist, script/style removal and six-hour
cache bound the integration. No resume or candidate text is sent to the job board.
