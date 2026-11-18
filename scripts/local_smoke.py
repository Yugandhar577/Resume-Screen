"""Explicit live local-model smoke test using fictional inputs only."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.local_review import review_locally
from ml.local_rewriter import generate_proposals, apply_local_edits

resume = 'Fictional Candidate\nExperience\nSoftware Engineer @ Example Labs (2023-2025)\n- Helped with Python APIs serving 50 users, not production\n- Worked on SQL reporting for 4 internal teams\nSkills\nPython SQL Git'
job = 'Software Engineer\nRequired: Python and SQL\nPreferred: Docker\nWrite readable APIs and tests.'
started = time.monotonic()
report = review_locally(resume, job)
print(json.dumps({'local_review': report['available'], 'model': report['model'], 'reviewers': len(report['reviewers']),
                  'alignment': len(report['alignment']), 'career_options': len(report['career_options']),
                  'discarded': report['rejected_unverified_items'], 'seconds': round(time.monotonic()-started, 2)}), flush=True)
started = time.monotonic()
rewrites = generate_proposals(resume, job)
keys = [e['id'] for e in rewrites['proposals']]
result = apply_local_edits(resume, job, keys, 'original', rewrites['proposalToken']) if rewrites['proposalToken'] else None
print(json.dumps({'rewrite_proposals': len(keys), 'rejected': rewrites['rejected'], 'accepted_edits_applied': result is not None,
                  'seconds': round(time.monotonic()-started, 2)}), flush=True)
