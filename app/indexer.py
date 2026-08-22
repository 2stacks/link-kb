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
import ipaddress
import json
import time
import logging
import requests
from urllib3.util import Retry
from urllib.parse import urlparse
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
STATUS_FILE = os.path.join(DB_PATH, "status.json")
# Link health tracking (report-only — never writes to Linkding)
HEALTH_FILE = os.path.join(DB_PATH, "link_health.json")
HEALTH_TRACKING = os.getenv("HEALTH_TRACKING", "1") != "0"
# Per-page fetch timeout for content/health fetches (seconds)
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "15"))
# Browser-like UA so plain-HTTP fetches don't trip naive bot walls
FETCH_USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_664; rv:130.0) Gecko/20100101 "
                    "Firefox/130.0 link-kb")

# Truncation limit — T4 nomic-embed has a 2048 token physical batch (-ub 2048).
# 2048 tokens ≈ 8000 chars. Leave headroom for title/tags/description (~200 chars).
MAX_CONTENT_LEN = 7000

# Query augmentation template — tells the embedding model to produce a
# retrieval-oriented vector rather than a definition/description vector.
QUERY_TEMPLATE = "Find bookmarks about: {query}"
# Per-request timeout for embedding calls (seconds). llama-swap can block
# for a long time while loading a model on a cold start; the readiness gate
# handles that, but a swap mid-run can also block.
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "120"))
# How long an index/search run will wait for the embeddings endpoint to
# become ready (cold-start model load). Probed every EMBEDDING_READY_INTERVAL s.
EMBEDDING_READY_TIMEOUT = int(os.getenv("EMBEDDING_READY_TIMEOUT", "300"))
EMBEDDING_READY_INTERVAL = 5


class EmbeddingError(Exception):
    """Raised when the embedding endpoint is unreachable or errors.

    Callers must NOT store a zero-vector fallback — a zero embedding
    silently destroys search relevance.
    """


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
        self._embed_failed = []
        self._health = {}
        self._load_status()
        self._load_health()
        # Persistent session with connection pooling to avoid CLOSE-WAIT buildup on llama-swap.
        # Generous connect/read retries with exponential backoff handle both stale pooled
        # connections (T4 server closes idle sockets) and mid-run llama-swap model swaps.
        # Cold-start unavailability is handled by _wait_for_embeddings() before runs start.
        self._session = requests.Session()
        retry = Retry(
            total=6, connect=6, read=5, backoff_factor=2,
            allowed_methods=frozenset({"GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"}),
        )
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=4, pool_connections=4, max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        # Separate session for page fetches: modest retries (a dead page should
        # fail fast, not burn a minute of backoff per link).
        self._fetch_session = requests.Session()
        fetch_retry = Retry(total=2, connect=2, read=1, backoff_factor=1,
                            allowed_methods=frozenset({"GET", "HEAD"}))
        fetch_adapter = requests.adapters.HTTPAdapter(pool_maxsize=4,
                                                      pool_connections=4,
                                                      max_retries=fetch_retry)
        self._fetch_session.mount("http://", fetch_adapter)
        self._fetch_session.mount("https://", fetch_adapter)

    def _load_status(self):
        """Load persisted timestamps from disk."""
        try:
            if os.path.exists(STATUS_FILE):
                with open(STATUS_FILE, "r") as f:
                    data = json.load(f)
                self._last_index_time = data.get("last_index_time")
                self._last_diff_index_time = data.get("last_diff_index_time")
        except Exception as e:
            logger.debug(f"Failed to load status: {e}")

    def _save_status(self):
        """Persist timestamps to disk."""
        try:
            os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
            with open(STATUS_FILE, "w") as f:
                json.dump({
                    "last_index_time": self._last_index_time,
                    "last_diff_index_time": self._last_diff_index_time,
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save status: {e}")

    def _load_health(self):
        """Load persisted link-health records from disk."""
        self._health = {}
        if not HEALTH_TRACKING:
            return
        try:
            if os.path.exists(HEALTH_FILE):
                with open(HEALTH_FILE, "r") as f:
                    self._health = json.load(f) or {}
        except Exception as e:
            logger.debug(f"Failed to load link health: {e}")
            self._health = {}

    def _save_health(self):
        """Persist link-health records to disk."""
        if not HEALTH_TRACKING:
            return
        try:
            os.makedirs(os.path.dirname(HEALTH_FILE), exist_ok=True)
            with open(HEALTH_FILE, "w") as f:
                json.dump(self._health, f)
        except Exception as e:
            logger.warning(f"Failed to save link health: {e}")

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
            "embed_failed_count": len(self._embed_failed),
            "link_health": self.get_health_summary(),
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
            raise EmbeddingError("all embedding chunks came back empty or zero")

        return self._single_embed(text)

    def _single_embed(self, text: str) -> list:
        """Embed a single text chunk.

        Raises EmbeddingError on failure — callers must NOT persist a
        zero-vector fallback (it silently destroys search relevance).
        """
        global EMBED_DIM
        payload = {
            "model": EMBEDDING_MODEL,
            "input": text
        }
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
            body = resp.text[:300] if resp is not None else "no response"
            raise EmbeddingError(f"{e} - {body}") from e

    def _probe_embeddings(self) -> bool:
        """Cheap liveness probe: POST one trivial token, expect a 200."""
        try:
            resp = self._session.post(
                self.embed_url,
                json={"model": EMBEDDING_MODEL, "input": "ping"},
                timeout=EMBEDDING_TIMEOUT,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Embedding probe failed: {e}")
            return False

    def _wait_for_embeddings(self, deadline_s: int = None, what: str = "") -> None:
        """Block until the embedding endpoint answers, for cold llama-swap starts.

        Raises EmbeddingError if the endpoint is still down after the deadline.
        """
        deadline_s = deadline_s if deadline_s is not None else EMBEDDING_READY_TIMEOUT
        if self._probe_embeddings():
            return
        label = f" for {what}" if what else ""
        logger.warning(
            f"Embeddings endpoint not responding{label} — waiting up to {deadline_s}s "
            f"(llama-swap cold start?). Probing every {EMBEDDING_READY_INTERVAL}s."
        )
        waited = 0
        while waited < deadline_s:
            time.sleep(EMBEDDING_READY_INTERVAL)
            waited += EMBEDDING_READY_INTERVAL
            if self._probe_embeddings():
                logger.info(f"Embeddings endpoint ready after {waited}s")
                return
            if waited % 30 == 0:  # periodic status, not per-probe spam
                logger.info(f"Still waiting for embeddings endpoint ({waited}s/{deadline_s}s)")
        raise EmbeddingError(
            f"Embeddings endpoint {self.embed_url} not ready after {deadline_s}s"
        )

    def _cleanup_poisoned_vectors(self) -> int:
        """Delete zero-embedding entries left behind by pre-v1.0.15 failure modes.

        Old code stored zero vectors when the embedding call failed, which
        silently destroys cosine similarity for those links. Returns the
        number of entries removed.
        """
        removed = 0
        for col_name, col in [("links_meta", self.meta_collection),
                              ("links_content", self.content_collection)]:
            try:
                data = col.get(include=["embeddings"])
            except Exception as e:
                logger.warning(f"Could not scan {col_name} for poisoned vectors: {e}")
                continue
            ids = data.get("ids") or []
            embeddings = data.get("embeddings")
            bad = []
            for doc_id, emb in zip(ids, embeddings if embeddings is not None else []):
                # emb is a list or numpy row — check for all-zero / empty
                try:
                    if emb is None or len(emb) == 0 or all(v == 0.0 for v in emb):
                        bad.append(doc_id)
                except Exception:
                    bad.append(doc_id)  # unreadable vector — safer to re-embed
            if bad:
                col.delete(ids=bad)
                removed += len(bad)
                logger.info(f"Removed {len(bad)} zero-embedding entries from {col_name}: {bad[:5]}{'...' if len(bad) > 5 else ''}")
        if removed:
            logger.info(f"Poisoned-vector cleanup: {removed} entries deleted — they will be re-embedded on this run")
        return removed

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
            time.sleep(0.1)
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

    def _classify_fetch(self, url: str, resp, err) -> dict:
        """Classify a page fetch for link-health tracking.

        Returns a record fragment: {class, status, reason, last_checked}.
        err: the exception from the GET (None when resp is not None).
        """
        rec: dict = {"last_checked": datetime.now(timezone.utc).isoformat()}

        if err is not None:
            # Unwrap the exception chain: urllib3/requests wrap the root
            # cause (e.g. read timeout) inside ConnectionError/MaxRetryError.
            chain, cause = [], err
            while cause is not None:
                chain.append((type(cause).__name__, str(cause)))
                cause = cause.__cause__ or cause.__context__
            names = " ".join(n for n, _ in chain)
            text = " ".join(t for _, t in chain).lower()
            if "timeout" in names or "timed out" in text:
                rec.update({"class": "suspect", "status": None, "reason": "timeout"})
            elif "SSLError" in names:
                rec.update({"class": "suspect", "status": None, "reason": "tls error"})
            elif "name" in text and ("resolve" in text or "getaddrinfo" in text):
                rec.update({"class": "suspect", "status": None, "reason": "dns failure"})
            else:
                rec.update({"class": "suspect", "status": None,
                            "reason": f"connection error: {chain[0][0]}"})
            return rec

        code = resp.status_code
        if 200 <= code < 400:
            rec.update({"class": "ok", "status": code, "reason": ""})
        elif code in (404, 410):
            rec.update({"class": "dead", "status": code, "reason": f"HTTP {code}"})
        elif code in (401, 402):
            # Auth/paywall: the resource exists, we just can't see it.
            rec.update({"class": "restricted", "status": code, "reason": f"HTTP {code}"})
        elif code == 405:
            # Method not allowed — server alive, page probably moved.
            rec.update({"class": "moved-suspect", "status": code, "reason": "HTTP 405"})
        else:
            rec.update({"class": "suspect", "status": code, "reason": f"HTTP {code}"})
        return rec

    def _update_health(self, link: dict, url: str, cls: str,
                       status, reason: str, final_url: str = ""):
        """Record/refresh a link's health record (streak logic)."""
        if not HEALTH_TRACKING:
            return
        lid = f"ld-{link.get('id')}"
        prev = self._health.get(lid, {})
        streak = prev.get("fail_streak", 0)
        if cls == "ok":
            streak = 0
        else:
            streak = streak + 1
        rec = {
            "class": cls,
            "status": status,
            "reason": reason,
            "last_checked": datetime.now(timezone.utc).isoformat(),
            "fail_streak": streak,
        }
        # Track stable redirect target (only when class is redirected)
        if cls == "redirected" and final_url:
            prev_target = prev.get("final_url")
            rec["final_url"] = final_url
            rec["redirect_streak"] = 1 if prev_target != final_url else prev.get("redirect_streak", 0) + 1
        self._health[lid] = rec

    def _prune_health(self, valid_ids: set):
        """Drop health records for links no longer in Linkding."""
        if not HEALTH_TRACKING:
            return
        stale = [k for k in self._health if k not in valid_ids]
        for k in stale:
            del self._health[k]
        if stale:
            logger.debug(f"Pruned {len(stale)} stale health records")

    def get_link_health(self, cls: str = None) -> list:
        """Return health records, optionally filtered by class (None = all)."""
        rows = []
        for lid, rec in self._health.items():
            if cls and rec.get("class") != cls:
                continue
            rows.append({"id": lid, **rec})
        # Worst first: highest fail streak, then most recent check
        rows.sort(key=lambda r: r.get("last_checked", ""), reverse=True)
        rows.sort(key=lambda r: (r.get("fail_streak") or 0), reverse=True)
        return rows

    def get_health_summary(self) -> dict:
        """Aggregate health counts for /api/status."""
        counts = {}
        for rec in self._health.values():
            c = rec.get("class", "unknown")
            counts[c] = counts.get(c, 0) + 1
        return {
            "tracked": len(self._health),
            "by_class": counts,
        }

    def _internal_host(self, url: str) -> str:
        """Return the host if it's a private/reserved IP literal (unreachable
        by design from outside the LAN), else ''."""
        try:
            host = (urlparse(url).hostname or "").strip("[]")
            if host:
                ip = ipaddress.ip_address(host)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    return host
        except (ValueError, Exception):
            pass
        return ""

    def _fetch_and_extract(self, url: str, link: dict) -> dict:
        """Fetch a page (recording health), then extract text with trafilatura.

        Fetching via our own session (browser-like UA, bounded timeout)
        gives us the HTTP status + final URL that trafilatura.fetch_url
        hides — needed for dead/moved classification.
        """
        # Intranet/private-IP links: skip the network call entirely —
        # they can't be reached from here, and hammering them is wasteful.
        internal = self._internal_host(url)
        if internal:
            self._update_health(link, url, "unreachable-internal", None,
                                f"private/reserved address {internal}")
            return {"title": "", "text": "", "final_url": url,
                    "health": {"class": "unreachable-internal", "status": None,
                               "reason": f"private/reserved address {internal}"}}

        rec = None
        html = None
        final_url = url
        try:
            resp = self._fetch_session.get(
                url, headers={"User-Agent": FETCH_USER_AGENT},
                timeout=FETCH_TIMEOUT, allow_redirects=True,
            )
            final_url = str(resp.url) if resp.url else url
            rec = self._classify_fetch(url, resp, None)
            if 200 <= resp.status_code < 400:
                ctype = resp.headers.get("Content-Type", "")
                if "text/html" in ctype or not ctype:
                    html = resp.text
                # non-HTML 2xx (PDF etc.): no extraction possible
        except Exception as e:
            rec = self._classify_fetch(url, None, e)

        # Redirected: final host/path differs from the bookmark URL
        if rec and rec["class"] == "ok" and final_url != url:
            a, b = urlparse(url), urlparse(final_url)
            if (a.netloc != b.netloc) or (a.path != b.path):
                rec = {**rec, "class": "redirected"}

        # Update health store
        self._update_health(
            link, url,
            rec.get("class", "suspect") if rec else "suspect",
            rec.get("status") if rec else None,
            rec.get("reason", "") if rec else "no record",
            final_url=final_url,
        )

        # Extract content from the fetched HTML. (Non-HTML 2xx such as PDFs
        # are not extracted — acceptable for the link corpus; it also avoids
        # a second network fetch that would double the load on slow links.)
        text = ""
        try:
            if html:
                text = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    include_links=False,
                    favor_precision=True,
                ) or ""
        except Exception as e:
            logger.debug(f"Extraction failed for {url}: {e}")

        return {
            "title": "",
            "text": text,
            "final_url": final_url,
            "health": rec or {"class": "suspect", "status": None, "reason": "no record"},
        }

    def _extract_page_content(self, url: str, link: dict = None) -> dict:
        """Fetch and extract text content from a URL, recording link health."""
        try:
            return self._fetch_and_extract(url, link or {"id": None, "url": url})
        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {e}")
            if link is not None:
                self._update_health(link, url, "suspect", None, f"fetch error: {type(e).__name__}")
            return {"title": "", "text": "", "final_url": url,
                    "health": {"class": "suspect", "status": None, "reason": str(e)}}

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

        self._wait_for_embeddings(what="full index")
        self._cleanup_poisoned_vectors()
        self._embed_failed = []

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

            # Extract page content (records link health)
            page_content = self._extract_page_content(url, link)

            # Build separate text for metadata and content embeddings
            meta_text = self._build_embedding_text(link, page_content)
            content_text = self._build_content_text(link, page_content)

            # Get embeddings — skip the link rather than storing a zero vector
            try:
                meta_vector = self._embed(meta_text)
                content_vector = self._embed(content_text)
            except EmbeddingError as e:
                logger.error(f"Embedding failed, skipping {url}: {e}")
                self._embed_failed.append(url)
                continue

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
        self._prune_health(expected_ids)
        self._save_status()
        self._save_health()
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

        self._wait_for_embeddings(what="diff index")
        self._cleanup_poisoned_vectors()
        self._embed_failed = []

        links = self._fetch_linkding_links()
        logger.info(f"Fetched {len(links)} links from Linkding")

        # Get current indexed IDs
        existing_ids = set(self.meta_collection.get()["ids"])
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

            page_content = self._extract_page_content(url, link)

            meta_text = self._build_embedding_text(link, page_content)
            content_text = self._build_content_text(link, page_content)

            try:
                meta_vector = self._embed(meta_text)
                content_vector = self._embed(content_text)
            except EmbeddingError as e:
                logger.error(f"Embedding failed, skipping {url}: {e}")
                self._embed_failed.append(url)
                continue

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
        self._prune_health(expected_ids)
        self._save_status()
        self._save_health()
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

        # Query augmentation — wrap in retrieval prompt. The readiness gate
        # keeps a cold llama-swap from silently producing a zero query vector
        # (which would return garbage rankings). Cap the wait below the
        # gunicorn worker timeout (300s) or the worker gets killed mid-request.
        self._wait_for_embeddings(
            deadline_s=min(EMBEDDING_READY_TIMEOUT, 240), what="search"
        )
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

  
