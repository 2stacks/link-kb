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
import urllib3
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
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "30"))


class Indexer:
    def __init__(self):
        self.headers = {
            "Authorization": f"Token {LINKDING_API_KEY}",
            "Accept": "application/json"
        }
        self.embed_url = f"{EMBEDDING_URL}/v1/embeddings"
        self.vector_store = None
        self._last_index_time = None
        self._last_diff_index_time = None
        self._total_indexed = 0
        # Persistent session with connection pooling to avoid CLOSE-WAIT buildup on llama-swap.
        # Retries on read-timeout handle stale pooled connections (T4 server closes idle sockets).
        self._session = requests.Session()
        retry = urllib3.util.Retry(total=5, read=4, backoff_factor=0.1)
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=4, pool_connections=4, max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def init_db(self):
        """Initialize ChromaDB persistent store with two collections."""
        os.makedirs(DB_PATH, exist_ok=True)
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        self.vector_store = chromadb.PersistentClient(path=DB_PATH)
        self.meta_collection = self.vector_store.get_or_create_collection(
            name="links_meta",
            metadata={"hnsw:space": "cosine"}
        )
        self.content_collection = self.vector_store.get_or_create_collection(
            name="links_content",
            metadata={"hnsw:space": "cosine"}
        )
        # Keep `self.collection` as alias for backward compat (points to meta)
        self.collection = self.meta_collection

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
        linkding_count = -1
        if self.vector_store:
            try:
                count = self.meta_collection.count()
            except Exception:
                pass
            try:
                linkding_count = self.get_linkding_count()
            except Exception:
                pass
        return {
            "total_indexed": count,
            "linkding_count": linkding_count,
            "last_index_time": self._last_index_time,
            "last_diff_index_time": self._last_diff_index_time,
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
            resp = self._session.post(self.embed_url, json=payload, timeout=EMBEDDING_TIMEOUT)
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
            resp = self._session.get(url, headers=self.headers, timeout=30)
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

    def get_linkding_count(self) -> int:
        """Get total bookmark count from Linkding (lightweight, limit=1)."""
        try:
            resp = self._session.get(
                f"{LINKDING_URL}/api/bookmarks/?limit=1&offset=0",
                headers=self.headers,
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json().get("count", 0)
        except Exception as e:
            logger.warning(f"Failed to get Linkding count: {e}")
        return -1

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
        """Build metadata text to embed from link title/tags/description/notes."""
        title = link.get("title", "")
        description = link.get("description", "")
        notes = link.get("notes", "")
        tag_names = link.get("tag_names", [])

        parts = []
        if title:
            parts.append(f"Title: {title}")
        if description:
            parts.append(f"Description: {description}")
        if notes:
            parts.append(f"Notes: {notes}")
        if tag_names:
            parts.append(f"Tags: {', '.join(tag_names)}")

        return "\n".join(parts)

    def _build_content_text(self, link: dict, page_content: dict) -> str:
        """Build page content text to embed separately."""
        page_text = page_content.get("text", "")
        url = link.get("url", "")

        parts = []
        if url:
            parts.append(f"URL: {url}")
        if page_text:
            parts.append(f"Content: {page_text}")

        result = "\n".join(parts)
        # Truncate to fit within model context window
        if len(result) > MAX_CONTENT_LEN + 200:
            result = result[:MAX_CONTENT_LEN + 200]
        return result

    def full_index(self, progress_callback=None) -> int:
        """Fetch all links from Linkding, embed them, and store.

        Removes entries for links no longer present in Linkding.

        Args:
            progress_callback: Optional callable(i, total, url) called after each link.
        """
        logger.info("Starting full index...")

        if not self.vector_store:
            self.init_db()

        links = self._fetch_linkding_links()
        logger.info(f"Fetched {len(links)} links from Linkding")

        # Determine which IDs should exist in the index
        expected_ids = {f"ld-{link.get('id')}" for link in links}

        # Remove stale entries (deleted in Linkding)
        for col_name, col in [("links_meta", self.meta_collection),
                              ("links_content", self.content_collection)]:
            all_ids = col.get(["ids"])["ids"]
            stale = [did for did in all_ids if did not in expected_ids]
            if stale:
                col.delete(ids=stale)
                logger.info(f"Removed {len(stale)} stale entries from {col_name}")

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

            # Build separate text for metadata and content embeddings
            meta_text = self._build_embedding_text(link, page_content)
            content_text = self._build_content_text(link, page_content)

            # Get embeddings
            meta_vector = self._embed(meta_text)
            content_vector = self._embed(content_text)

            # ChromaDB metadata
            metadata = {
                "url": url,
                "title": link.get("title", ""),
                "description": link.get("description", ""),
                "tags": "|".join(link.get("tag_names", [])),
                "date_added": link.get("date_added", ""),
                "date_modified": link.get("date_modified", ""),
            }

            # Store in both collections
            link_id = f"ld-{link.get('id', i)}"
            self.meta_collection.upsert(
                ids=[link_id],
                embeddings=[meta_vector],
                metadatas=[metadata],
                documents=[meta_text]
            )
            self.content_collection.upsert(
                ids=[link_id],
                embeddings=[content_vector],
                metadatas=[metadata],
                documents=[content_text]
            )

            # Small throttle to avoid overwhelming T4 llama-server
            time.sleep(0.1)

        self._last_index_time = datetime.now(timezone.utc).isoformat()
        self._total_indexed = len(links)
        logger.info(f"Index complete: {self._total_indexed} links stored")
        return self._total_indexed

    def diff_index(self, progress_callback=None) -> dict:
        """Incremental index — only process new or removed links.

        Compares current Linkding bookmarks against what's in ChromaDB.
        New links get extracted and embedded; removed links are deleted.
        Unchanged links are skipped entirely.

        Args:
            progress_callback: Optional callable(i, total, url) called after each link.

        Returns:
            dict with 'added', 'removed', 'unchanged' counts.
        """
        logger.info("Starting diff index...")

        if not self.vector_store:
            self.init_db()

        links = self._fetch_linkding_links()
        logger.info(f"Fetched {len(links)} links from Linkding")

        # Get current indexed IDs
        existing_ids = set(self.meta_collection.get(["ids"])["ids"])
        expected_ids = {f"ld-{link.get('id')}" for link in links}

        new_ids = expected_ids - existing_ids
        removed_ids = existing_ids - expected_ids

        logger.info(
            f"Diff index: {len(existing_ids)} existing, {len(expected_ids)} expected, "
            f"{len(new_ids)} new, {len(removed_ids)} removed"
        )

        added = 0
        removed = 0

        # Remove deleted links
        if removed_ids:
            for col_name, col in [("links_meta", self.meta_collection),
                                  ("links_content", self.content_collection)]:
                col.delete(ids=list(removed_ids))
            removed = len(removed_ids)
            logger.info(f"Removed {removed} stale entries")

        # Index new links
        new_links = [link for link in links
                     if f"ld-{link.get('id')}" in new_ids]
        unchanged = len(links) - len(new_links) - removed

        for i, link in enumerate(new_links):
            url = link.get("url", "")
            if not url:
                continue

            logger.info(f"Processing new [{i+1}/{len(new_links)}]: {url}")

            if progress_callback:
                try:
                    progress_callback(i + 1, len(new_links), url)
                except Exception:
                    pass

            page_content = self._extract_page_content(url)

            meta_text = self._build_embedding_text(link, page_content)
            content_text = self._build_content_text(link, page_content)

            meta_vector = self._embed(meta_text)
            content_vector = self._embed(content_text)

            metadata = {
                "url": url,
                "title": link.get("title", ""),
                "description": link.get("description", ""),
                "tags": "|".join(link.get("tag_names", [])),
                "date_added": link.get("date_added", ""),
                "date_modified": link.get("date_modified", ""),
            }

            link_id = f"ld-{link.get('id', i)}"
            self.meta_collection.upsert(
                ids=[link_id],
                embeddings=[meta_vector],
                metadatas=[metadata],
                documents=[meta_text]
            )
            self.content_collection.upsert(
                ids=[link_id],
                embeddings=[content_vector],
                metadatas=[metadata],
                documents=[content_text]
            )

            added += 1
            time.sleep(0.1)

        self._last_diff_index_time = datetime.now(timezone.utc).isoformat()
        logger.info(f"Diff index complete: {added} added, {removed} removed, {unchanged} unchanged")
        return {"added": added, "removed": removed, "unchanged": unchanged}

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
        """Two-vector semantic search: query metadata and content collections, merge by best score."""
        if not self.vector_store:
            self.init_db()

        # Query augmentation — wrap in retrieval prompt
        augmented_query = QUERY_TEMPLATE.format(query=query)
        query_vector = self._embed(augmented_query)

        # Fetch more than needed from each collection for merging
        fetch_limit = min(limit * 2, 40)

        # Query metadata collection
        meta_results = self.meta_collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_limit
        )

        # Query content collection
        content_results = self.content_collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_limit
        )

        # Merge results — keyed by link ID, keep best score
        merged = {}  # link_id -> {score, metadata}

        for src, results in [("meta", meta_results), ("content", content_results)]:
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            for doc_id, dist, meta in zip(ids, distances, metadatas):
                if meta is None:
                    continue
                # Cosine distance → similarity score
                score = max(0, 1 - (dist / 2)) if dist else 0

                if doc_id not in merged or score > merged[doc_id]["score"]:
                    merged[doc_id] = {
                        "url": meta.get("url", ""),
                        "title": meta.get("title", ""),
                        "description": meta.get("description", ""),
                        "tags": meta.get("tags", "").split("|") if meta.get("tags") else [],
                        "date_added": meta.get("date_added", ""),
                        "score": score,
                    }

        # Format and sort by best score
        formatted = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        # Apply keyword boost for short queries
        formatted = self._keyword_boost(query, formatted)

        # Trim to requested limit
        return formatted[:limit]

  
