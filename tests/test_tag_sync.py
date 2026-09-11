"""v1.0.27 — tag-sync: reconcile @HEALTH_HTTP_* tags with the health store.

Covers:
- _sync_health_tags() rules: add tag on 4xx/5xx (incl. drift 400->403),
  strip on recovery to 2xx, strip on unknown status (None), no-op when the
  tag already matches, archived bookmarks skipped, untracked links skipped,
  other tags always preserved.
- Gating: TAG_SYNC off -> zero Linkding requests, report enabled=False.
- Per-item error isolation: one 500 recorded, run continues.
- Wire-up: full_index() runs the sync at the end when gated on; a sync
  failure must never break the index.
- /api/status: tag_sync block + tag_sync_enabled present.
"""
import app.indexer as indexer_mod
from app.indexer import Indexer
from app import server


# --- fixtures ---------------------------------------------------------------

# Live bookmarks with various tag states.
BM = {
    201: {"id": 201, "url": "https://a.example/blocked", "tag_names": ["docs"]},
    202: {"id": 202, "url": "https://a.example/recovered", "tag_names": ["@HEALTH_HTTP_403", "training"]},
    203: {"id": 203, "url": "https://a.example/accurate", "tag_names": ["@HEALTH_HTTP_429", "apps"]},
    204: {"id": 204, "url": "https://a.example/drift", "tag_names": ["@HEALTH_HTTP_400", "docs"]},
    205: {"id": 205, "url": "https://a.example/unknown", "tag_names": ["@HEALTH_HTTP_502"]},
    206: {"id": 206, "url": "https://a.example/archived", "tag_names": ["@HEALTH_HTTP_404"]},
    207: {"id": 207, "url": "https://a.example/ok", "tag_names": ["tools"]},
    208: {"id": 208, "url": "https://a.example/untracked", "tag_names": ["@HEALTH_HTTP_404"]},
}
BM[206]["is_archived"] = True

HEALTH = {
    "ld-201": {"class": "restricted", "status": 403, "reason": "HTTP 403"},
    "ld-202": {"class": "ok", "status": 200, "reason": ""},          # recovered -> strip
    "ld-203": {"class": "suspect", "status": 429, "reason": "HTTP 429"},  # accurate -> no-op
    "ld-204": {"class": "restricted", "status": 403, "reason": "HTTP 403"},  # drift 400->403
    "ld-205": {"class": "suspect", "status": None, "reason": "timeout"},     # unknown -> strip
    "ld-206": {"class": "dead", "status": 404, "reason": "HTTP 404"},        # archived -> skip
    "ld-207": {"class": "ok", "status": 200, "reason": ""},
    # 208 has NO health record -> skipped entirely (never re-untag something
    # the store doesn't track).
}


def _ix():
    ix = Indexer()
    ix._health = dict(HEALTH)
    ix._fetch_linkding_links = lambda: [v for k, v in BM.items()]
    return ix


class _FakeResp:
    def __init__(self, code=200, text="ok"):
        self.status_code = code
        self.text = text


class _RecordSession:
    def __init__(self, codes=None):
        self.patches = []
        self.codes = codes or []
        self._n = 0

    def patch(self, url, headers=None, json=None, timeout=None):
        self.patches.append((url, json))
        code = self.codes[self._n] if self._n < len(self.codes) else 200
        self._n += 1
        return _FakeResp(code)


def _sent_by_bmid(sess):
    return {u.rsplit("/", 2)[-2]: j for u, j in sess.patches}


# --- rules -------------------------------------------------------------------

def test_sync_add_strip_and_drift(monkeypatch):
    monkeypatch.setattr(indexer_mod, "TAG_SYNC", True)
    ix = _ix()
    sess = _RecordSession()
    ix._session = sess
    rep = ix._sync_health_tags()
    sent = _sent_by_bmid(sess)
    # 201: 403, no tag yet -> add
    assert sent["201"] == {"tag_names": ["docs", "@HEALTH_HTTP_403"]}, sent["201"]
    # 202: recovered to 200 -> strip stale 403, keep 'training'
    assert sent["202"] == {"tag_names": ["training"]}, sent["202"]
    # 204: drift 400->403 -> tag corrected, 'docs' preserved
    assert sent["204"] == {"tag_names": ["docs", "@HEALTH_HTTP_403"]}, sent["204"]
    # 205: unknown status -> strip, no tag
    assert sent["205"] == {"tag_names": []}, sent["205"]
    # 203 accurate, 207 fine, 206 archived, 208 untracked -> NO writes
    assert "203" not in sent and "207" not in sent and "206" not in sent and "208" not in sent
    # 201,204 added; 202,205 removed; 203,207 already-correct -> unchanged
    assert rep["added"] == 2 and rep["removed"] == 2 and rep["unchanged"] == 2
    assert rep["failed"] == 0 and rep["enabled"] is True


def test_sync_accurate_is_noop_no_write(monkeypatch):
    monkeypatch.setattr(indexer_mod, "TAG_SYNC", True)
    ix = _ix()
    sess = _RecordSession()
    ix._session = sess
    rep = ix._sync_health_tags()
    # only 201,202,204,205 need writes; 203 (429) and 207 (ok) already correct
    assert len(sess.patches) == 4
    assert rep["unchanged"] == 2  # 203 + 207
    assert "@HEALTH_HTTP_429" in BM[203]["tag_names"]


def test_sync_gate_off_zero_requests(monkeypatch):
    monkeypatch.setattr(indexer_mod, "TAG_SYNC", False)
    ix = _ix()
    sess = _RecordSession()
    ix._session = sess
    rep = ix._sync_health_tags()
    assert sess.patches == []
    assert rep == {"enabled": False, "added": 0, "removed": 0, "unchanged": 0,
                   "failed": 0, "items": []}
    assert ix._last_tag_sync is None


def test_sync_error_isolation(monkeypatch):
    monkeypatch.setattr(indexer_mod, "TAG_SYNC", True)
    ix = _ix()
    # PATCH order: 201(add,200), 202(strip,500), 204(add,200), 205(strip,200)
    sess = _RecordSession(codes=[200, 500, 200, 200])
    ix._session = sess
    rep = ix._sync_health_tags()
    assert rep["failed"] == 1
    assert rep["added"] == 2 and rep["removed"] == 1  # 201,204 added; 205 removed; 202 failed
    bad = [i for i in rep["items"] if not i["ok"]]
    assert len(bad) == 1 and bad[0]["bm_id"] == 202
    # run completed all 4 items despite the one failure
    assert len(rep["items"]) == 4


# --- full_index wire-up --------------------------------------------------------
# (real implementations live after the _NoopColl / _full_index_ix helpers)


class _NoopColl:
    def upsert(self, *a, **k):
        pass

    def count(self):
        return 0

    def get(self, *a, **k):
        return {"ids": []}

    def delete(self, *a, **k):
        pass


def _full_index_ix(monkeypatch, tag_sync_on):
    """Indexer ready to run the real full_index() with zero heavy I/O.

    Stubs: vector store (truthy, skips init_db), embedding wait, poisoned-
    vector cleanup, no-op collections, and the tag-sync method under test.
    """
    monkeypatch.setattr(indexer_mod, "TAG_SYNC", tag_sync_on)
    ix = _ix()
    ix.vector_store = object()
    ix._wait_for_embeddings = lambda what="": None
    ix._cleanup_poisoned_vectors = lambda: 0
    ix.meta_collection = _NoopColl()
    ix.content_collection = _NoopColl()
    ix._fetch_linkding_links = lambda: []  # no links -> no per-link work
    return ix


def test_full_index_runs_tag_sync_when_on(monkeypatch):
    calls = {"n": 0}
    report = {"enabled": True, "added": 1, "removed": 0, "unchanged": 0,
              "failed": 0, "items": [], "at": "x"}
    ix = _full_index_ix(monkeypatch, True)

    def fake_sync():
        calls["n"] += 1
        ix._last_tag_sync = report  # mirrors the real method's assignment
        return report

    ix._sync_health_tags = fake_sync
    n = ix.full_index()
    assert n == 0
    assert calls["n"] == 1, "tag sync must run at end of full_index when gated on"
    assert ix._last_tag_sync == report


def test_full_index_survives_tag_sync_failure(monkeypatch):
    ix = _full_index_ix(monkeypatch, True)

    def boom():
        raise RuntimeError("linkding down")

    ix._sync_health_tags = boom
    n = ix.full_index()  # must NOT raise
    assert n == 0
    assert ix._last_tag_sync is None  # no partial report written


def test_full_index_skips_tag_sync_when_off(monkeypatch):
    calls = {"n": 0}
    ix = _full_index_ix(monkeypatch, False)
    ix._sync_health_tags = lambda: (calls.__setitem__("n", calls["n"] + 1), {})[1]
    n = ix.full_index()
    assert n == 0
    assert calls["n"] == 0, "tag sync must NOT run when gated off"


# --- /api/status ---------------------------------------------------------------

def test_status_exposes_tag_sync(monkeypatch):
    monkeypatch.setattr(indexer_mod, "TAG_SYNC", True)
    ix = _ix()
    ix._last_tag_sync = {"enabled": True, "added": 1, "removed": 1,
                         "unchanged": 1, "failed": 0, "items": [], "at": "now"}
    server.get_indexer = lambda: ix
    c = server.app.test_client()
    r = c.get("/api/status")
    assert r.status_code == 200
    body = r.get_json()
    assert body["tag_sync_enabled"] is True
    assert body["tag_sync"]["added"] == 1
