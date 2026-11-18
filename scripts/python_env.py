"""Use this project's virtual environment when launched by npm."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
python = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
executable = str(python) if python.is_file() else sys.executable
commands = {'backend': ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', os.getenv('BACKEND_PORT', '8001')],
            'test': ['-m', 'unittest', 'discover', '-s', 'tests', '-v']}
if len(sys.argv) < 2 or sys.argv[1] not in commands:
    raise SystemExit('Usage: python scripts/python_env.py backend|test')
raise SystemExit(subprocess.call([executable, *commands[sys.argv[1]], *sys.argv[2:]], cwd=root))
