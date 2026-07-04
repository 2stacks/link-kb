"""
Linkding indexer — fetches bookmarks, embeds content, stores vectors.

Uses trafilatura for page extraction and vecs (sqlite-vec) for
ANN search backed by llama-swap nomic-embed embeddings.
"""

import os
import time
import logging
import requests
import trafilatura
import vecs
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Config
LINKDING_URL = os.getenv("LINKDING_URL", "https://linkding.2stacks.net")
LINKDING_API_KEY = os.getenv("LINKDING_API_KEY", "")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8080")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
DB_PATH = os.getenv("DB_PATH", "/data/link-kb.db")
EMBED_DIM = 768  # nomic-embed-text-v1.5 dimension


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
        """Initialize vecs (sqlite-vec) store."""
        self.vector_store = vecs.VecClient(DB_PATH)
        # Create or get collection with 768-dim cosine index
        self.collection = self.vector_store.get_or_create_collection(
            "links",
            dimension=EMBED_DIM
        )

    def _embed(self, text: str) -> list:
        """Send text to llama-swap embedding endpoint."""
        payload = {
            "model": EMBEDDING_MODEL,
            "input": text
        }
        resp = None
        try:
            resp = requests.post(self.embed_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            body = resp.text if resp is not None else "no response"
            logger.error(f"Embedding failed: {e} - {body}")
            return [0.0] * EMBED_DIM  # fallback zero vector

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
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; LinkKB/1.0)"
            }
            content = trafilatura.fetch_url(url, timeout=15)
            if not content:
                return {"title": "", "text": ""}
            text = trafilatura.extract(
                content,
                include_comments=False,
                include_tables=True,
                include_links=False,
                favor_precision=True
            )
            # Get title from the HTML if possible
            try:
                title = trafilatura.extract(
                    content,
                    include_comments=False,
                    include_tables=False,
                    include_links=False,
                    favor_recall=False,
                    _extraction_kwargs={}
                )
            except Exception:
                title = ""
            return {"title": title or "", "text": text or ""}
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
        
        # Page content
        page_text = page_content.get("text", "")
        
        # Truncate page text to avoid embedding size limits
        max_content_len = 8000
        if len(page_text) > max_content_len:
            page_text = page_text[:max_content_len]
        
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

    def full_index(self) -> int:
        """Fetch all links from Linkding, embed them, and store."""
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
            
            # Extract page content
            page_content = self._extract_page_content(url)
            
            # Build text for embedding
            embed_text = self._build_embedding_text(link, page_content)
            
            # Get embedding
            vector = self._embed(embed_text)
            
            # Metadata for search results
            metadata = {
                "url": url,
                "title": link.get("title", ""),
                "description": link.get("description", ""),
                "tags": link.get("tag_names", []),
                "date_added": link.get("date_added", ""),
                "date_modified": link.get("date_modified", ""),
            }
            
            # Upsert into vecs (uses linkding ID as id)
            link_id = f"ld-{link.get('id', i)}"
            self.collection.upsert(
                rows=[(link_id, vector, metadata)]
            )
            
            # Rate limit to avoid hammering llama-swap
            time.sleep(0.5)
        
        self._last_index_time = datetime.now(timezone.utc).isoformat()
        self._total_indexed = len(links)
        logger.info(f"Index complete: {self._total_indexed} links stored")
        return self._total_indexed

    def search(self, query: str, limit: int = 10) -> list:
        """Semantic search: embed query, find nearest neighbors."""
        if not self.vector_store:
            self.init_db()
        
        query_vector = self._embed(query)
        
        # ANN search with cosine similarity
        results = self.collection.query(
            data=query_vector,
            namespace=None,
            limit=limit,
            measure="cosine_distance"
        )
        
        # Format results
        formatted = []
        for hit in results:
            metadata = hit.metadata
            formatted.append({
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "tags": metadata.get("tags", []),
                "date_added": metadata.get("date_added", ""),
                "score": float(hit.score),
            })
        
        return formatted

    def get_status(self) -> dict:
        """Return indexing status and stats."""
        count = 0
        if self.vector_store:
            count = self.collection.count()
        return {
            "total_indexed": count,
            "last_index_time": self._last_index_time,
            "linkding_url": LINKDING_URL,
            "embedding_model": EMBEDDING_MODEL,
            "db_path": DB_PATH,
        }
