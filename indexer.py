"""
Linkding indexer — fetches bookmarks, embeds content, stores vectors.

Uses trafilatura for page extraction and ChromaDB for
ANN search backed by llama-swap nomic-embed embeddings.

Search strategy:
  1. Query augmentation — wraps user query in a retrieval prompt template
     so the embedding model produces a better query vector.
  2. Keyword boost — for short queries (≤3 words), literally matches
     terms against title/tags/description and boosts those results.
"""

import os
import re
import time
import logging
import requests
import chromadb
import trafilatura
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Config
LINKDING_URL = os.getenv("LINKDING_URL", "https://linkding.2stacks.net")
LINKDING_API_KEY = os.getenv("LINKDING_API_KEY", "")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8080")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
DB_PATH = os.getenv("DB_PATH", "/data/link-kb")  # ChromaDB uses a directory
EMBED_DIM = None  # auto-detected from first embedding response

# Truncation limit — T4 nomic-embed has a 2048 token physical batch (-ub 2048).
# 2048 tokens ≈ 8000 chars. Leave headroom for title/tags/description (~200 chars).
MAX_CONTENT_LEN = 7000

# Query augmentation template — tells the embedding model to produce a
# retrieval-oriented vector rather than a definition/description vector.
QUERY_TEMPLATE = "Find bookmarks about: {query}"


class Indexer:
    def __init__(self):
        self.headers = {
            "Authorization": f"Token {LINKDING_API_KEY}",
            "Accept": "application/json"
        }
        self.embed_url = f"{EMBEDDING_URL}/v1/embeddings"
        self.vector_store = None
        self._last_index_time = None
        self._total_indexed = 0

    def init_db(self):
        """Initialize ChromaDB persistent store."""
        os.makedirs(DB_PATH, exist_ok=True)
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        self.vector_store = chromadb.PersistentClient(path=DB_PATH)
        # Always re-fetch the collection reference in case it was recreated
        self.collection = self.vector_store.get_or_create_collection(
            name="links",
            metadata={"hnsw:space": "cosine"}
        )

    def _ensure_collection(self):
        """Re-acquire collection reference if stale."""
        if not self.vector_store:
            self.init_db()
        else:
            try:
                self.collection = self.vector_store.get_collection("links")
            except Exception:
                self.collection = self.vector_store.get_or_create_collection(
                    name="links",
                    metadata={"hnsw:space": "cosine"}
                )

    def get_status(self) -> dict:
        """Return indexing status and stats."""
        count = 0
        self._ensure_collection()
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {
            "total_indexed": count,
            "last_index_time": self._last_index_time,
            "linkding_url": LINKDING_URL,
            "embedding_model": EMBEDDING_MODEL,
            "db_path": DB_PATH,
            "embedding_dim": EMBED_DIM,
        }

    def _embed(self, text: str) -> list:
        """Send text to llama-swap embedding endpoint.

        Chunks text to stay under the T4's 512 token physical batch limit.
        Returns the mean of chunk embeddings for long text.
        """
        global EMBED_DIM
        # Rough token estimate: 1 token ≈ 4 chars for English text
        # T4 nomic-embed has a 512 token physical batch limit
        est_tokens = len(text) // 4
        if est_tokens > 400:
            # Chunk into ~400 token pieces and average
            chunks = self._chunk_text(text)
            chunk_embs = []
            for chunk in chunks:
                emb = self._single_embed(chunk)
                if len(emb) and any(v != 0.0 for v in emb):
                    chunk_embs.append(emb)
            if chunk_embs:
                # Mean pooling across chunks
                dim = len(chunk_embs[0])
                mean_emb = [sum(e[i] for e in chunk_embs) / len(chunk_embs) for i in range(dim)]
                # Normalize
                norm = (sum(v*v for v in mean_emb) ** 0.5) or 1.0
                mean_emb = [v / norm for v in mean_emb]
                return mean_emb
            return [0.0] * (EMBED_DIM or 768)

        return self._single_embed(text)

    def _single_embed(self, text: str) -> list:
        """Embed a single text chunk, returning zero vector on failure."""
        global EMBED_DIM
        payload = {
            "model": EMBEDDING_MODEL,
            "input": text
        }
        resp = None
        try:
            resp = requests.post(self.embed_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            if EMBED_DIM is None:
                EMBED_DIM = len(embedding)
                logger.info(f"Detected embedding dim: {EMBED_DIM}")
            return embedding
        except Exception as e:
            body = resp.text if resp is not None else "no response"
            logger.error(f"Embedding failed: {e} - {body}")
            dim = EMBED_DIM or 768
            return [0.0] * dim

    def _chunk_text(self, text: str, max_chars: int = 800) -> list:
        """Split text into chunks that fit within token limits."""
        chunks = []
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i+max_chars])
        return chunks

    def _fetch_linkding_links(self) -> list:
        """Fetch all bookmarks from Linkding API."""
        links = []
        offset = 0
        limit = 100
        while True:
            url = f"{LINKDING_URL}/api/bookmarks/?limit={limit}&offset={offset}"
            resp = requests.get(url, headers=self.headers, timeout=30)
            if resp.status_code != 200:
                logger.error(f"Linkding API error: {resp.status_code} - {resp.text}")
                break
            data = resp.json()
            results = data.get("results", [])
            links.extend(results)
            if offset + len(results) >= data.get("count", 0):
                break
            offset += limit
        return links

    def _extract_page_content(self, url: str) -> dict:
        """Fetch and extract text content from a URL."""
        try:
            content = trafilatura.fetch_url(url)
            if not content:
                return {"title": "", "text": ""}
            text = trafilatura.extract(
                content,
                include_comments=False,
                include_tables=True,
                include_links=False,
                favor_precision=True
            )
            return {"title": "", "text": text or ""}
        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {e}")
            return {"title": "", "text": ""}

    def _build_embedding_text(self, link: dict, page_content: dict) -> str:
        """Build text to embed from link metadata + page content."""
        title = link.get("title", "")
        url = link.get("url", "")
        description = link.get("description", "")
        notes = link.get("notes", "")
        tag_names = link.get("tag_names", [])

        page_text = page_content.get("text", "")

        # Truncate page text to fit within model context window
        if len(page_text) > MAX_CONTENT_LEN:
            page_text = page_text[:MAX_CONTENT_LEN]

        parts = []
        if title:
            parts.append(f"Title: {title}")
        if description:
            parts.append(f"Description: {description}")
        if notes:
            parts.append(f"Notes: {notes}")
        if tag_names:
            parts.append(f"Tags: {', '.join(tag_names)}")
        parts.append(f"URL: {url}")
        if page_text:
            parts.append(f"Content: {page_text}")

        return "\n".join(parts)

    def full_index(self, progress_callback=None) -> int:
        """Fetch all links from Linkding, embed them, and store.

        Args:
            progress_callback: Optional callable(i, total, url) called after each link.
        """
        logger.info("Starting full index...")

        if not self.vector_store:
            self.init_db()

        links = self._fetch_linkding_links()
        logger.info(f"Fetched {len(links)} links from Linkding")

        for i, link in enumerate(links):
            url = link.get("url", "")
            if not url:
                continue

            logger.info(f"Processing [{i+1}/{len(links)}]: {url}")

            # Progress callback for real-time status
            if progress_callback:
                try:
                    progress_callback(i + 1, len(links), url)
                except Exception:
                    pass

            # Extract page content
            page_content = self._extract_page_content(url)

            # Build text for embedding
            embed_text = self._build_embedding_text(link, page_content)

            # Get embedding
            vector = self._embed(embed_text)

            # ChromaDB metadata
            metadata = {
                "url": url,
                "title": link.get("title", ""),
                "description": link.get("description", ""),
                "tags": "|".join(link.get("tag_names", [])),
                "date_added": link.get("date_added", ""),
                "date_modified": link.get("date_modified", ""),
            }

            # Store the embedded text as the document for debugging
            link_id = f"ld-{link.get('id', i)}"
            self.collection.upsert(
                ids=[link_id],
                embeddings=[vector],
                metadatas=[metadata],
                documents=[embed_text]
            )

            # Small throttle to avoid overwhelming T4 llama-server
            time.sleep(0.1)

        self._last_index_time = datetime.now(timezone.utc).isoformat()
        self._total_indexed = len(links)
        logger.info(f"Index complete: {self._total_indexed} links stored")
        return self._total_indexed

    def _keyword_boost(self, query: str, results: list) -> list:
        """
        Boost results that literally contain the query terms in title/tags/description.
        Only applies to short queries (≤3 words) where keyword matching is more useful.
        """
        query_words = re.split(r'\s+', query.lower().strip())

        # Don't boost for natural language queries (>3 words)
        if len(query_words) > 3:
            return results

        # Don't boost for empty or single-char queries
        if not query_words or all(len(w) <= 1 for w in query_words):
            return results

        def keyword_score(item: dict) -> float:
            """Return a boost factor based on how many query terms appear in metadata."""
            searchable = (
                item.get("title", "").lower() + " " +
                item.get("description", "").lower() + " " +
                " ".join(item.get("tags", [])).lower() + " " +
                item.get("url", "").lower()
            )
            matches = sum(1 for w in query_words if w in searchable)
            # Boost: 0.1 per matching word (max 0.3 for 3 words)
            return matches * 0.1

        # Apply boost to scores
        boosted = []
        for item in results:
            item["score"] = min(1.0, item["score"] + keyword_score(item))
            boosted.append(item)

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x["score"], reverse=True)
        return boosted

    def search(self, query: str, limit: int = 10) -> list:
        """Semantic search with query augmentation + keyword boost."""
        if not self.vector_store:
            self.init_db()

        # Query augmentation — wrap in retrieval prompt
        augmented_query = QUERY_TEMPLATE.format(query=query)
        query_vector = self._embed(augmented_query)

        # ChromaDB query — fetch a bit more than needed for keyword boosting
        fetch_limit = min(limit * 3, 50)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_limit
        )

        # Format results
        formatted = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc_id, dist, meta in zip(ids, distances, metadatas):
            if meta is None:
                continue
            tags = meta.get("tags", "").split("|") if meta.get("tags") else []
            # Cosine distance → similarity score
            score = max(0, 1 - (dist / 2)) if dist else 0
            formatted.append({
                "url": meta.get("url", ""),
                "title": meta.get("title", ""),
                "description": meta.get("description", ""),
                "tags": tags,
                "date_added": meta.get("date_added", ""),
                "score": round(score, 4),
            })

        # Apply keyword boost for short queries
        formatted = self._keyword_boost(query, formatted)

        # Trim to requested limit
        return formatted[:limit]

  
