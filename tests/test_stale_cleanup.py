"""v1.0.19 regression: full_index() must delete entries whose bookmark no
longer exists in Linkding — from BOTH collections.

Pre-fix, existing ids were listed with col.get(["ids"]). In chromadb the
first positional argument of Collection.get() is an `ids` FILTER, not an
include list, so the call fetched the (nonexistent) id "ids" and silently
returned []. stale was always empty and the cleanup deleted nothing —
deleted bookmarks only ever left the index via diff_index.
"""
from app.indexer import Indexer

VEC = [0.1, 0.2, 0.3]

# Current Linkding state: ld-2 was deleted after the previous index.
LIVE_LINKS = [
    {"id": 1, "url": "https://a.example/1", "title": "A1", "description": "d1", "tag_names": []},
    {"id": 3, "url": "https://a.example/3", "title": "A3", "description": "d3", "tag_names": []},
    {"id": 4, "url": "https://a.example/4", "title": "A4", "description": "d4", "tag_names": []},
]
ALL_SEEDED = ["ld-1", "ld-2", "ld-3", "ld-4"]


def _patched_indexer():
    """Indexer with all I/O monkeypatched — runs fully offline."""
    ix = Indexer()
    ix.init_db()
    ix._wait_for_embeddings = lambda deadline_s=None, what="": None
    ix._cleanup_poisoned_vectors = lambda: 0
    ix._fetch_linkding_links = lambda: LIVE_LINKS
    ix._extract_page_content = lambda url, link=None: {"text": f"page {url}", "url": url}
    ix._embed = lambda text: VEC
    ix._probe_endpoint = lambda: "up"
    ix._breaker_check = lambda *a, **k: (0, False)
    return ix


def _seed(ix, ids):
    """Seed both collections as if a previous full index stored `ids`."""
    for link_id in ids:
        url = f"https://a.example/{link_id.split('-')[1]}"
        meta = {"url": url, "title": "T", "description": "D", "tags": "",
                "date_added": "", "date_modified": ""}
        ix.meta_collection.upsert(ids=[link_id], embeddings=[VEC],
                                  metadatas=[meta], documents=[f"meta {url}"])
        ix.content_collection.upsert(ids=[link_id], embeddings=[VEC],
                                     metadatas=[meta], documents=[f"content {url}"])


def test_full_index_deletes_stale_from_both_collections():
    ix = _patched_indexer()
    _seed(ix, ALL_SEEDED)
    assert ix.meta_collection.count() == 4
    assert ix.content_collection.count() == 4

    total = ix.full_index()

    assert total == len(LIVE_LINKS)
    meta_ids = sorted(ix.meta_collection.get(include=["metadatas"])["ids"])
    content_ids = sorted(ix.content_collection.get(include=["metadatas"])["ids"])
    assert meta_ids == ["ld-1", "ld-3", "ld-4"], f"stale entry survived links_meta: {meta_ids}"
    assert content_ids == ["ld-1", "ld-3", "ld-4"], f"stale entry survived links_content: {content_ids}"


def test_full_index_is_idempotent_for_live_links():
    ix = _patched_indexer()
    _seed(ix, ALL_SEEDED)
    ix.full_index()
    total = ix.full_index()

    assert total == len(LIVE_LINKS)
    assert ix.meta_collection.count() == len(LIVE_LINKS)
    assert ix.content_collection.count() == len(LIVE_LINKS)
