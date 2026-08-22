# link-kb

Semantic knowledge base for your saved bookmarks. Natural language search over Linkding links using `nomic-embed` vector embeddings and **ChromaDB** ANN search.

## What it does

- **Indexes** all your Linkding bookmarks — fetches page content via `trafilatura`, embeds with your chosen embedding model
- **Searches** with natural language — "that tool for monitoring Kubernetes" finds relevant links by semantic similarity
- **Tracks link health** — every indexed page is classified (ok / dead / moved / blocked / unreachable) with a failure streak, so you can find and prune dead or relocated bookmarks. Report-only: it never writes back to Linkding.
- **Serves** a minimal web UI for search, plus a REST API

## Embedding service

The indexer needs an embedding model endpoint compatible with the OpenAI API format
(`POST /v1/embeddings` with `model` and `input` fields). Any of these work:

- **OpenAI** — `text-embedding-3-small` (1536d)
- **Ollama** — `nomic-embed-text`, `bge-m3`, etc.
- **llama.cpp / llama-swap** — self-hosted GGUF embedding models
- **OpenRouter, Together, etc.** — any provider with embeddings support

Set `EMBEDDING_URL` and `EMBEDDING_MODEL` in `.env` to match your provider.

**Note:** Content is truncated to 7000 chars per link. Ensure your model's context
window can handle that (≥2048 tokens). For self-hosted llama.cpp, use `-ub 2048` or
higher on the embedding server.

## Architecture

```
Linkding API ──▶ indexer ──▶ Embedding model ──▶ ChromaDB
                      ▲
               Flask API + Web UI
```

## Quick start

```bash
# 1. Copy and fill environment
cp .env.example .env
# Edit .env with your Linkding API key and embedding service URL

# 2. Run locally (needs embedding service on localhost:8080)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py

# 3. Run in Docker
docker compose up -d --build

# 4. Trigger a re-index
curl -X POST http://localhost:5000/api/full-index
```

## Configuration

| Variable | Description | Default |
|---|---|---|
| `LINKDING_URL` | Linkding base URL | `https://linkding.2stacks.net` |
| `LINKDING_API_KEY` | Linkding API token | (required) |
| | `EMBEDDING_URL` | Embedding service base URL | `http://localhost:8080` |
| `EMBEDDING_MODEL` | Embedding model ID | `nomic-embed-text-v1.5` |
| `DB_PATH` | ChromaDB directory | `/data/link-kb` |
| `EMBEDDING_TIMEOUT` | Per-embed-request timeout (s) | `120` |
| `EMBEDDING_READY_TIMEOUT` | Cold-start wait for embeddings endpoint (s) | `300` |
| `HEALTH_TRACKING` | Track link health during indexing (`1`/`0`) | `1` |
| `FETCH_TIMEOUT` | Per-page fetch timeout (s) | `15` |

## Link health

Every page fetched during indexing is classified and stored in
`link_health.json` (under `DB_PATH`):

| Class | Meaning |
|---|---|
| `ok` | Fetched, HTTP 2xx/3xx |
| `dead` | HTTP 404/410 |
| `redirected` | Moved to a new URL (`final_url` + `redirect_streak` recorded) |
| `restricted` | HTTP 401/402 — exists, but auth/paywalled |
| `moved-suspect` | HTTP 405 — server alive, page likely moved |
| `suspect` | 403/429/5xx, timeout, TLS, DNS — may be transient |
| `unreachable-internal` | Private/reserved IP — unreachable by design, skipped |

`fail_streak` increments on each non-ok check and resets on ok, so a
one-off hiccup is distinguishable from a persistently dead link. Records
prune automatically when links are deleted in Linkding.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/search?q=...` | GET | Semantic search (param: `limit`, default 10) |
| `/api/full-index` | POST | Trigger full re-index |
| `/api/diff-index` | POST | Trigger incremental index |
| `/api/status` | GET | Indexing stats + link-health summary |
| `/api/link-health` | GET | Health records (param: `class` filter), worst-first |

## Deploy

1. Update `.env` with your embedding service URL and Linkding API credentials
2. `docker compose up -d --build`
3. Trigger initial index: `curl -X POST http://localhost:5000/api/full-index`

## Stack

- **Flask** — API + web UI
- **ChromaDB** — Persistent vector DB (SQLite + HNSW backend)
- **trafilatura** — Clean page content extraction
- **Embedding model** — Configurable via `EMBEDDING_MODEL` (default: nomic-embed-text-v1.5, 768d)
- **Gunicorn** — Production WSGI server