"""Flask server — routes and background indexing."""

import os
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template

from .indexer import Indexer

app = Flask(__name__, template_folder="../templates")

# Lazy-init indexer
indexer = None
index_lock = threading.Lock()
_indexing = False
_index_progress = {"total": 0, "processed": 0, "started_at": None, "done": False}


def get_indexer():
    global indexer
    if indexer is None:
        indexer = Indexer()
        indexer.init_db()
    return indexer


def _run_index():
    """Run full index in background thread."""
    global _indexing
    try:
        ix = get_indexer()
        def on_progress(i, total, url):
            with index_lock:
                _index_progress["processed"] = i
                _index_progress["total"] = total
                _index_progress["started_at"] = _index_progress.get("started_at") or datetime.now().isoformat()
                _index_progress["current_url"] = url
        total = ix.full_index(progress_callback=on_progress)
        with index_lock:
            _index_progress["total"] = total
            _index_progress["processed"] = total
            _index_progress["started_at"] = _index_progress.get("started_at") or datetime.now().isoformat()
            _index_progress["done"] = True
    except Exception as e:
        with index_lock:
            _index_progress["done"] = True
            _index_progress["error"] = str(e)
    finally:
        _indexing = False


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
    results = ix.search(query, limit=limit)

    return jsonify({
        "query": query,
        "count": len(results),
        "results": results
    })


@app.route("/api/index", methods=["POST"])
def trigger_index():
    """Start a full re-index in the background."""
    global _indexing
    with index_lock:
        if _indexing:
            return jsonify({
                "status": "already_running",
                "progress": _index_progress
            }), 409
        _indexing = True
        _index_progress = {"total": 0, "processed": 0, "started_at": None, "done": False}
    t = threading.Thread(target=_run_index, daemon=True)
    t.start()
    return jsonify({
        "status": "started",
        "message": "Indexing in background. Check /api/status for progress."
    })


@app.route("/api/status")
def status():
    """Indexing status and stats."""
    ix = get_indexer()
    with index_lock:
        progress = dict(_index_progress)
    base = ix.get_status()
    base["indexing"] = _indexing
    base["progress"] = progress
    return jsonify(base)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
