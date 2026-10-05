"""
Entry point to run the entire AI STEM Ecosystem application (Backend + Frontend).

Usage:
  python run.py
Or:
  uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_VENV_PY = _ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")


def _running_in_project_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == _VENV_PY.resolve()
    except OSError:
        return False


if _VENV_PY.is_file() and not _running_in_project_venv():
    # Bare `python run.py` on Windows often hits Microsoft Store Python, which
    # does not have beanie / the rest of requirements.txt.
    os.execv(str(_VENV_PY), [str(_VENV_PY), *sys.argv])

import uvicorn

if __name__ == "__main__":
    print("Starting AI STEM Ecosystem on http://127.0.0.1:8000 ...")
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
