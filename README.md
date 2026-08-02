# link-kb

Semantic knowledge base for your saved bookmarks. Natural language search over Linkding links using `nomic-embed` vector embeddings and **ChromaDB** ANN search.

## What it does

- **Indexes** all your Linkding bookmarks — fetches page content via `trafilatura`, embeds with your chosen embedding model
- **Searches** with natural language — "that tool for monitoring Kubernetes" finds relevant links by semantic similarity
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
curl -X POST http://localhost:5000/api/index
```

## Configuration

| Variable | Description | Default |
|---|---|---|
| `LINKDING_URL` | Linkding base URL | `https://linkding.2stacks.net` |
| `LINKDING_API_KEY` | Linkding API token | (required) |
|| `EMBEDDING_URL` | Embedding service base URL | `http://localhost:8080` |
| `EMBEDDING_MODEL` | Embedding model ID | `nomic-embed-text-v1.5` |
| `DB_PATH` | ChromaDB directory | `/data/link-kb` |

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/search?q=...` | GET | Semantic search (param: `limit`, default 10) |
| `/api/index` | POST | Trigger full re-index |
| `/api/status` | GET | Indexing stats |

## Deploy

1. Update `.env` with your embedding service URL and Linkding API credentials
2. `docker compose up -d --build`
3. Trigger initial index: `curl -X POST http://localhost:5000/api/index`

## Stack

- **Flask** — API + web UI
- **ChromaDB** — Persistent vector DB (SQLite + HNSW backend)
- **trafilatura** — Clean page content extraction
- **Embedding model** — Configurable via `EMBEDDING_MODEL` (default: nomic-embed-text-v1.5, 768d)
- **Gunicorn** — Production WSGI server