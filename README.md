# link-kb

Semantic knowledge base for your saved bookmarks. Natural language search over Linkding links using `nomic-embed` vector embeddings and `sqlite-vec` ANN search.

## What it does

- **Indexes** all your Linkding bookmarks — fetches page content via `trafilatura`, embeds with nomic-embed via llama-swap
- **Searches** with natural language — "that tool for monitoring Kubernetes" finds relevant links by semantic similarity
- **Serves** a minimal web UI for search, plus a REST API

## Architecture

```
Linkding API ──▶ indexer ──▶ llama-swap (nomic-embed) ──▶ sqlite-vec (vecs)
                      ▲
               Flask API + Web UI
```

## Quick start

```bash
# 1. Copy and fill environment
cp .env.example .env
# Edit .env with your Linkding API key and llama-swap URL

# 2. Run locally (needs llama-swap on localhost:8080)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py

# 3. Run in Docker
docker compose up -d --build

# 4. Trigger a re-index
curl -X POST http://localhost:5000/api/index
```

## Configuration

| Variable | Description | Default |
|---|---|---|
| `LINKDING_URL` | Linkding base URL | `https://linkding.2stacks.net` |
| `LINKDING_API_KEY` | Linkding API key | (required) |
| `EMBEDDING_URL` | llama-swap base URL | `http://localhost:8080` |
| `EMBEDDING_MODEL` | Embedding model ID | `nomic-embed-text-v1.5` |
| `DB_PATH` | SQLite vector DB path | `/data/link-kb.db` |

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/search?q=...` | GET | Semantic search (param: `limit`, default 10) |
| `/api/index` | POST | Trigger full re-index |
| `/api/status` | GET | Indexing stats |

## Deploy

1. Copy project to the linkding host
2. Update `.env` — set `EMBEDDING_URL=http://host.docker.internal:8080` (production)
3. `docker compose up -d --build`
4. Add Caddy config (see `Caddyfile.example`)
5. Trigger initial index: `curl -X POST http://localhost:5000/api/index`

## Stack

- **Flask** — API + web UI
- **vecs** — Python wrapper for sqlite-vec (ANN search in SQLite)
- **trafilatura** — Clean page content extraction
- **nomic-embed-text-v1.5** — 768-dim embeddings via llama-swap
- **Gunicorn** — Production WSGI server
