"""Shared canonical vocabulary for the app and research data preparation."""

import re
from functools import lru_cache

SKILL_ALIASES = {
    "Python": [], "Java": [], "JavaScript": ["JS"], "TypeScript": ["TS"],
    "C++": ["CPP"], "C#": ["C sharp"], "Go": ["Golang"], "Rust": [], "R": [],
    "PHP": [], "Ruby": [], "Swift": [], "Kotlin": [], "Scala": [], "MATLAB": [],
    "Bash": [], "SQL": [], "HTML": [], "CSS": [], "React": ["React.js", "ReactJS"],
    "Node.js": ["NodeJS", "Node JS"], "Next.js": ["NextJS"], "Angular": [], "Vue": ["Vue.js"],
    "Django": [], "Flask": [], "FastAPI": [], "Spring": ["Spring Boot"], "Express": ["Express.js"],
    "GraphQL": [], "REST APIs": ["REST", "REST API", "RESTful APIs"], "Microservices": [],
    "MongoDB": [], "PostgreSQL": ["Postgres"], "MySQL": [], "Redis": [], "SQLite": [],
    "Elasticsearch": [], "DynamoDB": [], "Oracle": [], "SQL Server": [],
    "AWS": ["Amazon Web Services"], "Azure": ["Microsoft Azure"], "GCP": ["Google Cloud", "Google Cloud Platform"],
    "Docker": [], "Kubernetes": ["K8s"], "Terraform": [], "Ansible": [], "Linux": [],
    "Git": [], "GitHub": [], "GitLab": [], "Jenkins": [], "GitHub Actions": [],
    "CI/CD": ["CI CD", "continuous integration", "continuous delivery"], "DevOps": [], "Cloud": [],
    "Machine Learning": ["ML"], "Artificial Intelligence": ["AI"], "Deep Learning": [],
    "Data Science": [], "NLP": ["Natural Language Processing"], "Computer Vision": [],
    "Graph Neural Networks": ["GNN", "GNNs"], "MLOps": [], "TensorFlow": [], "PyTorch": [],
    "scikit-learn": ["sklearn"], "Pandas": [], "NumPy": [], "Keras": [], "XGBoost": [],
    "Spark": ["Apache Spark", "PySpark"], "Hadoop": [], "Tableau": [], "Power BI": [],
    "Excel": ["Microsoft Excel"], "Statistics": ["statistical analysis"], "A/B Testing": [],
    "Agile": [], "Scrum": [], "Figma": [], "UX": ["user experience"], "UI": ["user interface"],
    "Project Management": [], "Leadership": [], "Communication": [], "Customer Service": [],
    "Accounting": [], "Bookkeeping": [], "Patient Care": [], "Teaching": [],
}


@lru_cache(maxsize=256)
def _pattern(alias):
    return re.compile(r"(?<![\w+#])" + re.escape(alias) + r"(?![\w+#])", re.I)


def is_negated(text, start, end):
    line_start = text.rfind('\n', 0, start) + 1
    line_end = text.find('\n', end)
    line_end = len(text) if line_end < 0 else line_end
    prefix = text[line_start:start].lower().replace('not only', '')
    prefix = re.split(r'[.;:]|\bbut\b|\bhowever\b', prefix)[-1]
    suffix = text[end:line_end].lower()
    return bool(re.search(r'\b(?:no|not|without|lack|lacking|never)\b[^.;:]{0,65}$', prefix)
                or re.match(r'\s+(?:is |experience is )?(?:not required|not needed|not proficient)', suffix))


def skill_evidence(text):
    evidence = {}
    for skill, aliases in SKILL_ALIASES.items():
        matches = []
        for alias in [skill, *aliases]:
            for match in _pattern(alias).finditer(text):
                # Lowercase 'go' and 'r' are usually ordinary words, not skill names.
                if alias in ("Go", "R") and match.group() != alias:
                    continue
                if is_negated(text, match.start(), match.end()):
                    continue
                matches.append(match)
        if matches:
            lines = []
            for match in sorted(matches, key=lambda m: m.start()):
                start = text.rfind("\n", 0, match.start()) + 1
                end = text.find("\n", match.end())
                line = text[start:end if end >= 0 else len(text)].strip()
                if line not in lines:
                    lines.append(line)
            evidence[skill] = lines
    return dict(sorted(evidence.items()))


def extract_skills(text):
    return list(skill_evidence(text))
