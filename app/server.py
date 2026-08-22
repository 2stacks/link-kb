"""Flask server — routes and background indexing."""

import os
import sys
import logging
import threading
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from apscheduler.schedulers.background import BackgroundScheduler

from .indexer import Indexer, EmbeddingError

app = Flask(__name__, template_folder="../templates")

# Logging config
_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

# Ensure chromadb logger respects LOG_LEVEL
logging.getLogger("chromadb").setLevel(_log_level)
# Silence chromadb telemetry — known posthog version mismatch, not actionable
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL + 1)

# Background scheduler for periodic indexing
scheduler = BackgroundScheduler(daemon=True)

# Lazy-init indexer
indexer = None
index_lock = threading.Lock()
_full_indexing = False
_diff_indexing = False
_full_progress = {"total": 0, "processed": 0, "started_at": None, "done": False}
_diff_progress = {"total": 0, "processed": 0, "started_at": None, "done": False}


def get_indexer():
    global indexer
    if indexer is None:
        indexer = Indexer()
        indexer.init_db()
    return indexer


def _run_full_index():
    """Run full index in background thread."""
    global _full_indexing
    try:
        ix = get_indexer()
        def on_progress(i, total, url):
            with index_lock:
                _full_progress["processed"] = i
                _full_progress["total"] = total
                _full_progress["started_at"] = _full_progress.get("started_at") or datetime.now().isoformat()
                _full_progress["current_url"] = url
        total = ix.full_index(progress_callback=on_progress)
        with index_lock:
            _full_progress["total"] = total
            _full_progress["processed"] = total
            _full_progress["started_at"] = _full_progress.get("started_at") or datetime.now().isoformat()
            _full_progress["done"] = True
    except Exception as e:
        with index_lock:
            _full_progress["done"] = True
            _full_progress["error"] = str(e)
    finally:
        _full_indexing = False


def _run_diff_index():
    """Run diff index in background thread."""
    global _diff_indexing
    try:
        ix = get_indexer()
        def on_progress(i, total, url):
            with index_lock:
                _diff_progress["processed"] = i
                _diff_progress["total"] = total
                _diff_progress["started_at"] = _diff_progress.get("started_at") or datetime.now().isoformat()
                _diff_progress["current_url"] = url
        result = ix.diff_index(progress_callback=on_progress)
        with index_lock:
            _diff_progress["total"] = result.get("added", 0)
            _diff_progress["processed"] = result.get("added", 0)
            _diff_progress["started_at"] = _diff_progress.get("started_at") or datetime.now().isoformat()
            _diff_progress["diff_result"] = result
            _diff_progress["done"] = True
    except Exception as e:
        with index_lock:
            _diff_progress["done"] = True
            _diff_progress["error"] = str(e)
    finally:
        _diff_indexing = False


@app.route("/")
def index_page():
    """Serve the search UI."""
    return render_template("index.html")


@app.route("/api/search")
def search():
    """
    Semantic search endpoint.

    Query params:
      q: Natural language query string
      limit: Max results (default 10)
    """
    query = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 10))

    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    ix = get_indexer()
    try:
        results = ix.search(query, limit=limit)
    except EmbeddingError as e:
        return jsonify({
            "error": f"Embedding service unavailable: {e}"
        }), 503

    return jsonify({
        "query": query,
        "count": len(results),
        "results": results
    })


@app.route("/api/full-index", methods=["POST"])
def trigger_full_index():
    """Start a full re-index in the background."""
    global _full_indexing
    with index_lock:
        if _full_indexing:
            return jsonify({
                "status": "already_running",
                "progress": _full_progress
            }), 409
        _full_indexing = True
        _full_progress = {"total": 0, "processed": 0, "started_at": None, "done": False}
    t = threading.Thread(target=_run_full_index, daemon=True)
    t.start()
    return jsonify({
        "status": "started",
        "message": "Full indexing in background. Check /api/status for progress."
    })


@app.route("/api/diff-index", methods=["POST"])
def trigger_diff_index():
    """Start a diff re-index in the background."""
    global _diff_indexing
    with index_lock:
        if _diff_indexing:
            return jsonify({
                "status": "already_running",
                "progress": _diff_progress
            }), 409
        _diff_indexing = True
        _diff_progress = {"total": 0, "processed": 0, "started_at": None, "done": False}
    t = threading.Thread(target=_run_diff_index, daemon=True)
    t.start()
    return jsonify({
        "status": "started",
        "message": "Diff indexing in background. Check /api/status for progress."
    })


@app.route("/api/status")
def status():
    """Indexing status and stats."""
    ix = get_indexer()
    with index_lock:
        full_progress = dict(_full_progress)
        diff_progress = dict(_diff_progress)
    base = ix.get_status()
    base["full_indexing"] = _full_indexing
    base["diff_indexing"] = _diff_indexing
    base["full_progress"] = full_progress
    base["diff_progress"] = diff_progress
    # Backward compat aliases — default to whichever is active
    if _full_indexing:
        base["indexing"] = True
        base["progress"] = full_progress
    elif _diff_indexing:
        base["indexing"] = True
        base["progress"] = diff_progress
    else:
        base["indexing"] = False
        base["progress"] = diff_progress if diff_progress.get("done") else full_progress
    return jsonify(base)


@app.route("/api/link-health")
def link_health():
    """Link health report (read-only).

    Query params:
      class: Filter by health class (dead|suspect|redirected|restricted|
             moved-suspect|unreachable-internal|ok). Omit for all.
    Returns records sorted worst-first (fail streak desc, then last_checked).
    """
    cls = request.args.get("class", "").strip() or None
    ix = get_indexer()
    rows = ix.get_link_health(cls)
    return jsonify({
        "count": len(rows),
        "class": cls,
        "records": rows,
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)


# --- Scheduled indexing (runs under gunicorn too) ---

def _schedule_full_index():
    """Start full indexer on schedule."""
    global _full_indexing
    with index_lock:
        if _full_indexing:
            return
        _full_indexing = True
        _full_progress = {"total": 0, "processed": 0, "started_at": None, "done": False}
    t = threading.Thread(target=_run_full_index, daemon=True)
    t.start()


def _schedule_diff_index():
    """Start diff indexer on schedule."""
    global _diff_indexing
    with index_lock:
        if _diff_indexing or _full_indexing:
            return
        _diff_indexing = True
        _diff_progress = {"total": 0, "processed": 0, "started_at": None, "done": False}
    t = threading.Thread(target=_run_diff_index, daemon=True)
    t.start()


# Full index interval — 0 means disabled (default: 0 = manual only)
_full_interval = int(os.getenv("FULL_INDEX_INTERVAL_HOURS", "0"))
if _full_interval > 0:
    scheduler.add_job(
        _schedule_full_index,
        trigger="interval",
        hours=_full_interval,
        max_instances=1,
        coalesce=True,
        id="full_index",
        name="full_index",
    )
    logger_f = __import__("logging").getLogger(__name__)
    logger_f.info(f"Full index scheduled every {_full_interval}h")

# Diff index interval — 0 means disabled (default: 24h)
_diff_interval = int(os.getenv("DIFF_INDEX_INTERVAL_HOURS", "24"))
if _diff_interval > 0:
    scheduler.add_job(
        _schedule_diff_index,
        trigger="interval",
        hours=_diff_interval,
        max_instances=1,
        coalesce=True,
        id="diff_index",
        name="diff_index",
    )
    logger_d = __import__("logging").getLogger(__name__)
    logger_d.info(f"Diff index scheduled every {_diff_interval}h")

scheduler.start()
