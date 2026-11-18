"""Explicit synthetic-data experiments; local weights only, never production scores."""
import re
from threading import Lock

from ml.model_registry import artifact_status

_lock = Lock()
_bert = None
_t5 = None


def jd_experiment(text):
    status = artifact_status('ml/models/jd_bias', 'job_bias')
    if not status.get('trained'):
        return {'available': False, 'reason': status['reason']}
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    global _bert
    with _lock, torch.inference_mode():
        torch.set_num_threads(4)
        if _bert is None:
            path = 'ml/models/jd_bias'
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).eval()
            _bert = tokenizer, model
        tokenizer, model = _bert
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 510:
            return {'available': False, 'reason': 'BERT experiment is limited to 510 tokens; no text was silently omitted.'}
        def activation(value):
            encoded = tokenizer(value, return_tensors='pt')
            return torch.sigmoid(model(**encoded).logits)[0].tolist()
        original = activation(text)
        phrases = list(dict.fromkeys(match.group().strip() for match in re.finditer(r'[^.!?\n]+[.!?]?', text) if match.group().strip()))[:12]
        removals = [(phrase, activation(text.replace(phrase, '', 1) or ' ')) for phrase in phrases]
        labels = status['provenance']['supported_labels']
        findings = []
        for i, label in enumerate(labels):
            threshold = status['provenance']['evaluation']['val']['labels'][label]['validation_threshold']
            evidence = sorted([{'quote': phrase, 'activation_drop': round(original[i]-scores[i], 6)} for phrase, scores in removals], key=lambda row: -row['activation_drop'])[:3]
            findings.append({'category': label, 'activation': round(original[i], 6), 'validation_threshold': threshold,
                             'flagged': original[i] >= threshold, 'sentence_removal': evidence})
        from ml.weak_wording import analyze
        return {'available': True, 'version': status['version'], 'labels': findings, 'weak_supervision':analyze(text),
                'note': 'BERT trained on 60 AI-reviewed fictional descriptions. Activations are uncalibrated; sentence removal measures this model’s sensitivity, not proof of bias or causation. At most 12 distinct sentences tested.',
                'provenance': status['provenance']}


def t5_candidates(bullets, job):
    status = artifact_status('ml/models/t5_rewriter', 'resume_rewriting')
    if not status.get('trained'):
        raise ValueError(status['reason'])
    global _t5
    with _lock:
        import torch
        torch.set_num_threads(4)
        if _t5 is None:
            from ml.rewriter.t5_rewriter import T5ResumeRewriter
            _t5 = T5ResumeRewriter(model_path='ml/models/t5_rewriter')
        edits = []
        for bullet in bullets[:4]:
            prompt = f"rewrite resume bullet for job: {job} original: {bullet['text']}"
            if len(_t5.tokenizer.encode(prompt)) > 256:
                continue
            result = _t5.rewrite_bullet(bullet['text'], job, 1)[0]['rewritten']
            edits.append({'line': bullet['line'], 'original': bullet['text'], 'proposed': result,
                          'reason': 'Experimental T5 proposal from AI-reviewed fictional training pairs; protected-fact and Gemma comparison checks required.'})
        return {'edits': edits}
