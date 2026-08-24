"""Run the suite WITHOUT pytest: python tests/run_all.py

Usual path is `python -m pytest tests/ -v`. This runner exists so the
suite can also be executed inside the shipped image, where pytest is not
installed by default:

    docker run --rm --entrypoint python \
        -v "$PWD/tests:/app/tests" \
        ghcr.io/2stacks/link-kb:latest \
        /app/tests/run_all.py
"""
import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conftest  # noqa: F401  env setup must run before app imports

MODULES = ["test_stale_cleanup", "test_progress_state"]

failures = 0
for mod_name in MODULES:
    mod = importlib.import_module(mod_name)
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        try:
            getattr(mod, name)()
            print(f"PASS {mod_name}.{name}")
        except Exception:
            failures += 1
            print(f"FAIL {mod_name}.{name}")
            traceback.print_exc()

print()
if failures:
    print(f"RESULT: {failures} failure(s)")
    sys.exit(1)
print("RESULT: all tests passed")
