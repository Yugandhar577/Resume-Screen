"""Conservative formatting that preserves resume content and verifies every token."""

from collections import Counter
import re


class ResumeOptimizer:
    def prepare_resume(self, original: str):
        lines = []
        changes = set()
        for raw in original.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = raw.strip()
            if line != raw:
                changes.add("Removed leading and trailing whitespace")
            if re.match(r"^[•●▪*]\s+", line):
                line = re.sub(r"^[•●▪*]\s+", "- ", line)
                changes.add("Standardized bullet markers")
            if not line and (not lines or not lines[-1]):
                changes.add("Removed redundant blank lines")
                continue
            lines.append(line)
        result = "\n".join(lines).strip()
        validation = self.validate_optimization(original, result)
        if not validation["valid"]:
            raise ValueError("Resume formatting failed the content-preservation check.")
        return {"resume": result, "method": "Formatting_Only", "changes": sorted(changes),
                "validation": validation,
                "note": "Formatting cleanup only. Skills, experience, achievements, and wording are preserved."}

    def optimize_resume(self, original_resume: str, job_description: str, analysis_insights: dict) -> str:
        return self.prepare_resume(original_resume)["resume"]

    @staticmethod
    def _content_tokens(text):
        lines = [re.sub(r"^\s*[-•●▪*]\s+", "", line) for line in text.splitlines()]
        return Counter(re.findall(r"\w+|[^\w\s]", "\n".join(lines).casefold()))

    def validate_optimization(self, original: str, optimized: str) -> dict:
        before = self._content_tokens(original)
        after = self._content_tokens(optimized)
        added = sorted((after - before).elements())
        removed = sorted((before - after).elements())
        return {"valid": not added and not removed,
                "issues": (["Content was added or removed"] if added or removed else []),
                "added_tokens": added, "removed_tokens": removed,
                "scope": "Token preservation for formatting-only transformations; not a semantic validator."}
