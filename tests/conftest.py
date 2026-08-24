"""Pytest bootstrap — isolated environment BEFORE any app.* import.

- Fresh temp DB_PATH per test session (ChromaDB persistent store)
- Health tracking off, index schedulers disabled (no background jobs in tests)
- Repo root on sys.path so `app.*` imports resolve

Every test must stay offline: Linkding fetches, page extraction, and
embedding calls are monkeypatched per module. Never add a test that
needs network or a real model endpoint.
"""
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["DB_PATH"] = tempfile.mkdtemp(prefix="linkkb_test_")
os.environ["HEALTH_TRACKING"] = "0"
os.environ["FULL_INDEX_INTERVAL_HOURS"] = "0"
os.environ["DIFF_INDEX_INTERVAL_HOURS"] = "0"

sys.path.insert(0, _REPO_ROOT)
