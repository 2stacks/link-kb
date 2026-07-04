"""
link-kb — Semantic knowledge base for your saved links.

Natural language search over Linkding bookmarks using nomic-embed
vector embeddings and sqlite-vec ANN search.
"""

import os
from flask import Flask, request, jsonify, render_template
from indexer import Indexer

app = Flask(__name__)

# Lazy-init indexer
indexer = None


def get_indexer():
    global indexer
    if indexer is None:
        indexer = Indexer()
        indexer.init_db()
    return indexer


@app.route("/")
def index():
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
    """Manually trigger a full re-index."""
    ix = get_indexer()
    total = ix.full_index()
    return jsonify({
        "status": "completed",
        "links_indexed": total
    })


@app.route("/api/status")
def status():
    """Indexing status and stats."""
    ix = get_indexer()
    return jsonify(ix.get_status())


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
