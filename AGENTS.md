# AGENTS.md — working rules for AI agents in this repo

Applies to any coding agent (Hermes, Claude Code, Codex, …) working in this
repository. Human contributors: this is also the project's release checklist.

## Release process

Release = a version bump. In this order:

1. Bump `.version` (keep the trailing newline).
2. Commit with message `vX.Y.Z - <one-line summary>`.
3. Tag `vX.Y.Z`.
4. Code/behavior change: `docker build -t ghcr.io/2stacks/link-kb:latest -t ghcr.io/2stacks/link-kb:vX.Y.Z .`
5. **Verify the new code is actually inside the container BEFORE pushing.**
   Builds are layer-cached and fast — a fast build can still be a stale one.
   Check the changed function/line in the image, e.g.
   `docker run --rm --entrypoint python ghcr.io/2stacks/link-kb:vX.Y.Z -c "..."`
   (the ENTRYPOINT is gunicorn; always pass `--entrypoint`).
6. Push the image (`latest` + version tag), then `git push origin main --tags`.
7. Docs-only change: steps 1-3 + git push only; no image rebuild.

The remote deployment is pull-based by the user — agents never deploy.

## Documentation must track the code (verify before tagging)

The README drifted badly once; don't let it drift again. Before tagging any
release that touches configuration, API, or behavior:

- **README config table**: must cover **every** `os.getenv(...)` in `app/`.
  Verify with a script, not by eye — diff the code's env vars against the
  table rows and require both sets to match exactly (no missing, no invented).
- **`.env.example`**: no dead variables — every var listed must be read by
  code. New optional vars go in as commented lines with their default value
  and a one-line purpose comment.
- **Provider-agnostic docs**: never tie documentation to a specific host,
  hardware, or service (no machine names, no GPU models, no host-specific
  URLs). Users choose their own embedding backend and deployment.
- New API endpoints get a row in the README API table in the same commit.

## Tests

- Suite lives in `tests/` (pytest style, all offline — Linkding fetches,
  page extraction, and embedding calls are monkeypatched; never add a test
  that needs network or a real model endpoint).
- Run with `python tests/run_all.py` (works with or without pytest — it's
  the standard runner for this repo; the shipped image has no pytest).
  With pytest installed, `python -m pytest tests/ -v` works too.
- Run against the shipped image too, since the image pins its own dependency
  versions: `docker run --rm --entrypoint python ghcr.io/2stacks/link-kb:latest /app/tests/run_all.py`
- **Every bug fix ships with a regression test that fails on the pre-fix
  commit and passes after.** Verify both directions before releasing.

## Lessons learned (bugs that shipped before being caught)

- **chromadb `Collection.get()` positional arg trap.** `col.get(["ids"])` is
  an `ids` FILTER, not an include list — it silently returns `[]` for a
  nonexistent id. Always pass `include=` as a keyword. (This made the
  full-index stale cleanup a silent no-op for four releases; deleted
  bookmarks only left the index via diff_index. Regression:
  `tests/test_stale_cleanup.py`.)
- **Python `global` rebinding trap.** Reassigning a module-level name inside
  a function without declaring `global` creates a throwaway local; module
  state silently keeps its old value. (This hid UI progress on every index
  run after the first. Regression: `tests/test_progress_state.py`.)
- **Silent no-ops are the dangerous class.** Both bugs above failed without
  any error or log line. When writing cleanup/delete logic that depends on a
  list of ids, assert the list is non-trivial (or log its length) in dev.

## Known limitations (not bugs)

- Single gunicorn worker: in-memory index progress (`full_progress` /
  `diff_progress`) is per-process and is lost on container restart. The UI
  then shows no progress bar until the next run — expected behavior.
- After a completed full index, the `Links:` count in the UI should equal
  Linkding's API count. If it doesn't, suspect the bookmark fetch, not the
  cleanup.
