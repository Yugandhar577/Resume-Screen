# AI review record

At the project owner's explicit request, Codex read and reviewed all 24 rewriting
drafts and all 60 job-description drafts. Every prepared example was approved for
an explicitly synthetic training experiment. This is AI review, not human review.

- Rewriting: the targets only replace `Helped with` with `Supported` or `Worked on`
  with `Contributed to`. Source identity, numbers, technologies, qualifiers, negation
  and responsibility levels remain present. No candidate experience was invented.
- JD labels: checked each supplied definition against the complete description.
  Vague labels correspond to explicit unspecified/unbounded expectations; exclusionary
  wording labels identify stated age/gender/language restrictions needing context
  review; experience inflation labels identify a stated entry-level/no-experience
  role conflicting with a 12-year requirement. These are not legal/fairness findings.
- Approval decisions and rationales are in `ai-review-decisions.json`; approved CSVs
  and checksum-bound provenance are in `ai-approved/`.
- Every resulting provenance has `review_type=ai`, `ai_reviewed=true`, and
  `reviewed=false`. Training requires the explicit synthetic experiment flag.

The small examples repeat phrase templates across source-disjoint splits. Held-out
results measure this exercise, not real-world generalization. No web upload or real
candidate is part of this data. Users may inspect and replace the decisions at `/review`.
