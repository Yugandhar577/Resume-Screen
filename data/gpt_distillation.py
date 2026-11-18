"""Legacy module name retained for imports; teacher generation now uses local Gemma."""
from ml.local_rewriter import generate_proposals

class LocalDistillationLabeller:
    """Draft pairs only. A human must review them before training."""
    def improve_bullet(self, original_bullet, jd_context=''):
        source = original_bullet if original_bullet.startswith(('-', '•', '*')) else '- '+original_bullet
        result = generate_proposals(source, jd_context or 'Improve wording while retaining every fact.')
        edits = [e for e in result['proposals'] if e['id'].startswith('local-')]
        return edits[0]['proposed'] if edits else source

    def generate_training_pairs(self, resume_bullets, batch_size=50):
        return [(bullet, self.improve_bullet(bullet)) for bullet in resume_bullets if len(bullet.strip()) > 10]

GPTDistillationLabeller = LocalDistillationLabeller
