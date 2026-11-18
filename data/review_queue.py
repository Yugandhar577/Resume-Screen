"""Explicit source-file review queues; web uploads never enter this pipeline."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, columns):
    target = Path(path)
    if target.exists():
        raise ValueError('Output already exists. Choose a new path to preserve previous review work.')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def source_metadata(path):
    metadata = json.loads(Path(path).read_text(encoding='utf-8'))
    if not metadata.get('source_url', '').strip() or not metadata.get('license', '').strip():
        raise ValueError('Source URL and usage license are required before creating a review queue.')
    return metadata


def seed(input_path, provenance, output_path, task, draft=False):
    metadata = source_metadata(provenance)
    rows = read_csv(input_path)
    if not rows:
        raise ValueError('Input contains no records.')
    fields = ['source_id', 'original', 'improved', 'jd_context'] if task == 'resume_rewriting' else ['source_id', 'text', *metadata.get('labels', [])]
    if task == 'job_bias' and (not metadata.get('labels') or not metadata.get('label_definitions')):
        raise ValueError('JD queues require labels and documented label_definitions.')
    if Path(output_path).with_suffix('.source.json').exists():
        raise ValueError('Source provenance output exists. Choose a new output path.')
    prepared = []
    for row in rows:
        if not row.get('source_id', '').strip() or not row.get('original' if task == 'resume_rewriting' else 'text', '').strip():
            raise ValueError('Every record requires source_id and source text.')
        record = {field: row.get(field, '') for field in fields}
        if draft and task == 'resume_rewriting':
            from data.gpt_distillation import LocalDistillationLabeller
            record['improved'] = LocalDistillationLabeller().improve_bullet(record['original'], record['jd_context'])
        record.update(review_status='pending', reviewer='', review_note='')
        prepared.append(record)
    write_csv(output_path, prepared, [*fields, 'review_status', 'reviewer', 'review_note'])
    Path(output_path).with_suffix('.source.json').write_text(json.dumps({**metadata, 'task': task, 'reviewed': False,
        'input_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        'label_method': 'Local teacher drafts awaiting human review' if draft else 'Awaiting human review'}, indent=2), encoding='utf-8')
    return len(prepared)


def export_reviewed(queue_path, output_path, allow_ai_reviewed_synthetic=False):
    queue = Path(queue_path)
    metadata = source_metadata(queue.with_suffix('.source.json'))
    rows = read_csv(queue)
    if any(row.get('review_status') not in ('approved', 'rejected') for row in rows):
        raise ValueError('Every queue record needs an explicit approved or rejected decision.')
    if any(not row.get('reviewer', '').strip() for row in rows):
        raise ValueError('Every decision requires a named human reviewer attestation.')
    accepted = [row for row in rows if row['review_status'] == 'approved']
    ai_review = any(row.get('review_type') == 'ai' for row in accepted)
    if any(row.get('review_type') not in (None, '', 'ai', 'human') for row in rows):
        raise ValueError('Unknown review type.')
    if ai_review and (not allow_ai_reviewed_synthetic or metadata.get('synthetic') is not True):
        raise ValueError('A dataset containing AI approvals requires an explicitly selected synthetic experiment.')
    if not accepted:
        raise ValueError('No approved records.')
    task = metadata['task']
    fields = ['source_id', 'original', 'improved', 'jd_context'] if task == 'resume_rewriting' else ['source_id', 'text', *metadata['labels']]
    for row in accepted:
        if task == 'resume_rewriting' and not row.get('improved', '').strip():
            raise ValueError('Approved rewriting records require a reviewed target.')
        if task == 'job_bias' and any(row.get(label) not in ('0', '1') for label in metadata['labels']):
            raise ValueError('Approved JD labels must all be reviewed binary 0 or 1 values.')
    output = Path(output_path)
    if output.with_suffix('.provenance.json').exists():
        raise ValueError('A provenance output already exists. Choose a fresh output path.')
    write_csv(output, [{field: row[field] for field in fields} for row in accepted], fields)
    manifest = {**metadata, 'reviewed': not ai_review, 'review_type': 'ai' if ai_review else 'human',
                'ai_reviewed': ai_review, 'reviewer': ', '.join(sorted({row['reviewer'] for row in accepted})),
                'label_method': 'Explicit AI review of fictional targets/labels' if ai_review else 'Explicit human acceptance of individually reviewed targets/labels',
                'data_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'review_queue_sha256': hashlib.sha256(queue.read_bytes()).hexdigest(),
                'approved_review_counts': {'ai':sum(row.get('review_type') == 'ai' for row in accepted),
                                           'human':sum(row.get('review_type') != 'ai' for row in accepted)},
                'review_attestation': 'Reviewer names are supplied attestations, not independently verified identities.'}
    output.with_suffix('.provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return len(accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('seed')
    prepare.add_argument('--input', required=True); prepare.add_argument('--provenance', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--task', choices=['resume_rewriting', 'job_bias'], required=True)
    prepare.add_argument('--local-drafts', action='store_true')
    export = commands.add_parser('export')
    export.add_argument('--queue', required=True); export.add_argument('--output', required=True)
    imported = commands.add_parser('import-decisions')
    imported.add_argument('--bundle', required=True); imported.add_argument('--output', required=True)
    imported.add_argument('--allow-ai-reviewed-synthetic', action='store_true')
    args = parser.parse_args()
    if args.command == 'seed':
        count = seed(args.input, args.provenance, args.output, args.task, args.local_drafts)
        print(f'Created {count} pending records. Review the CSV and set review_status, reviewer, and review_note.')
    elif args.command == 'export':
        count = export_reviewed(args.queue, args.output)
        print(f'Exported {count} reviewed records with a checksum-bound provenance manifest.')
    else:
        bundle = json.loads(Path(args.bundle).read_text(encoding='utf-8'))
        if bundle.get('version') != 1 or not bundle.get('source', {}).get('synthetic'):
            raise ValueError('Expected a fictional review-workspace export.')
        output = Path(args.output)
        for name, task in [('rewrites','resume_rewriting'),('jobs','job_bias')]:
            rows = bundle[name]
            queue = output/(name+'-queue.csv')
            write_csv(queue, rows, list(rows[0]))
            queue.with_suffix('.source.json').write_text(json.dumps({**bundle['source'],'task':task,
                'labels':list(bundle['label_definitions']),'label_definitions':bundle['label_definitions']},indent=2),encoding='utf-8')
            if all(row.get('review_status') in ('approved','rejected') for row in rows) and any(row.get('review_status') == 'approved' for row in rows):
                export_reviewed(queue,output/(name+'-reviewed.csv'),args.allow_ai_reviewed_synthetic)
        print('Imported decisions. Pending records remain unavailable for training.')


if __name__ == '__main__':
    main()
