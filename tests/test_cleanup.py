"""v1.0.23 — Phase 3 dead-link cleanup: plan rules + gated write-back.

Covers:
- get_cleanup_plan(): rule selection (dead streak/410, redirect streak),
  skips (already archived, wrong streak, non-redirected classes),
  URL sanitizer (utm_*/session junk stripped, benign params kept),
  scope filtering.
- apply_cleanup(): exact PATCH payloads, per-item error isolation.
- /api/check-links: GET dry-run, POST refused (403) when the
  LINK_HEALTH_AUTO_ARCHIVE gate is off, 200 with writes when on,
  400 on bad scope.
"""
import app.indexer as indexer_mod
from app.indexer import Indexer
from app import server


# --- fixtures -----------------------------------------------------------------

BM = {
    101: {"id": 101, "url": "https://a.example/gone", "description": "", "tag_names": []},
    102: {"id": 102, "url": "https://a.example/old410", "description": "", "tag_names": []},
    103: {"id": 103, "url": "https://a.example/one", "description": "", "tag_names": []},
    104: {"id": 104, "url": "https://a.example/moved", "description": "", "tag_names": []},
    105: {"id": 105, "url": "https://a.example/tracked", "description": "", "tag_names": []},
    106: {"id": 106, "url": "https://a.example/ok", "description": "", "tag_names": []},
    107: {"id": 107, "url": "https://a.example/archived", "description": "", "tag_names": []},
    108: {"id": 108, "url": "https://a.example/was-archived", "description": "", "tag_names": []},
    109: {"id": 109, "url": "https://a.example/desc", "description": "keep me", "tag_names": []},
}
BM[107]["is_archived"] = True
BM[108]["is_archived"] = True
# 110: a LIVE bookmark already sitting at a redirect destination
BM[110] = {"id": 110, "url": "https://b.example/moved", "description": "", "tag_names": []}
# 111: a stale bookmark whose redirect target is the already-bookmarked 110
BM[111] = {"id": 111, "url": "https://a.example/old-moved", "description": "", "tag_names": []}

HEALTH = {
    "ld-101": {"class": "dead", "status": 404, "reason": "HTTP 404", "fail_streak": 3},
    "ld-102": {"class": "dead", "status": 410, "reason": "HTTP 410", "fail_streak": 1},
    "ld-103": {"class": "dead", "status": 404, "reason": "HTTP 404", "fail_streak": 1},
    "ld-104": {"class": "redirected", "status": 200, "reason": "",
               "final_url": "https://b.example/new", "redirect_streak": 3},
    "ld-105": {"class": "redirected", "status": 200, "reason": "",
               "final_url": "https://b.example/new?utm_source=mail&utm_campaign=x&keep=1",
               "redirect_streak": 3},
    "ld-106": {"class": "ok", "status": 200, "reason": "", "fail_streak": 0},
    "ld-107": {"class": "dead", "status": 404, "reason": "HTTP 404", "fail_streak": 5},
    "ld-108": {"class": "redirected", "status": 200, "reason": "",
               "final_url": "https://b.example/x", "redirect_streak": 4},
    "ld-109": {"class": "redirected", "status": 200, "reason": "",
               "final_url": "https://b.example/desc", "redirect_streak": 2},
    # 111: redirect target is ALREADY bookmarked (110) → must be skipped,
    # never proposed as an update (would create a duplicate URL)
    "ld-111": {"class": "redirected", "status": 200, "reason": "",
               "final_url": "https://b.example/moved", "redirect_streak": 3},
}


def _ix(health=None, bms=None):
    ix = Indexer()
    ix._health = dict(health if health is not None else HEALTH)
    ix._fetch_linkding_links = lambda: [v for k, v in (bms or BM).items()]
    return ix


# --- get_cleanup_plan ------------------------------------------------------------

def _by_action(plan, action):
    return {p["bm_id"]: p for p in plan["planned"] if p["action"] == action}


def test_plan_rules():
    ix = _ix()
    plan = ix.get_cleanup_plan()
    arch = _by_action(plan, "archive")
    upd = _by_action(plan, "update_url")
    # dead: streak >= 2 → 101; single 410 → 102; streak 1 non-410 → 103 excluded
    assert set(arch) == {101, 102}, f"archive set wrong: {sorted(arch)}"
    # redirect: streak >= 2 with stable final → 104, 105, 109
    assert set(upd) == {104, 105, 109}, f"update set wrong: {sorted(upd)}"
    # never touched: ok, dead-already-archived(107), redirected-already-archived(108)
    assert plan["counts"] == {"archive": 2, "update_url": 3}


def test_sanitizer_strips_tracking_keeps_benign():
    ix = _ix()
    plan = ix.get_cleanup_plan(scope="redirects")
    upd = _by_action(plan, "update_url")
    assert upd[105]["final"] == "https://b.example/new?keep=1", upd[105]["final"]
    # direct unit checks
    assert ix._sanitize_final_url("https://x.example/a;jsessionid=abc") == \
        "https://x.example/a;jsessionid=abc"  # path junk left alone (streak protects)
    assert ix._sanitize_final_url("https://x.example/a?utm_source=y&ref=z") == \
        "https://x.example/a"
    assert ix._sanitize_final_url("https://x.example/a?keep=1") == "https://x.example/a?keep=1"


def test_plan_skips_dup_targets_without_hiding_them():
    # v1.0.24: a redirect whose destination is already a live bookmark must
    # NOT be proposed as update_url (would create a duplicate URL), but it is
    # surfaced in skipped_dup_targets — never a silent no-op.
    ix = _ix()
    plan = ix.get_cleanup_plan()
    upd = _by_action(plan, "update_url")
    assert 111 not in upd, f"dup-target leak into plan: {[p['original'] for p in plan['planned']]}"
    assert plan["skipped_dup_count"] == 1
    skip = plan["skipped_dup_targets"][0]
    assert skip["bm_id"] == 111 and skip["final"] == "https://b.example/moved"
    # the plan's 'counts' only counts actionable items
    assert plan["counts"] == {"archive": 2, "update_url": 3}


def test_plan_description_flag_and_scope_filter():
    ix = _ix()
    plan = ix.get_cleanup_plan(scope="redirects")
    upd = _by_action(plan, "update_url")
    # 109 has an existing description → flag set, note must NOT be clobbered
    assert upd[109]["has_description"] is True
    assert upd[104]["has_description"] is False
    # scope=redirects filters out archives
    assert all(p["action"] == "update_url" for p in plan["planned"])
    plan_a = ix.get_cleanup_plan(scope="archive")
    assert all(p["action"] == "archive" for p in plan_a["planned"])


# --- apply_cleanup ----------------------------------------------------------------

class _FakeResp:
    def __init__(self, code=200, text="ok"):
        self.status_code = code
        self.text = text


def test_apply_archive_payloads():
    ix = _ix()
    calls = []

    def fake_patch(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        return _FakeResp(200)

    ix._session.patch = fake_patch
    plan = ix.get_cleanup_plan(scope="archive")
    result = ix.apply_cleanup(plan, scope="archive")
    assert result["applied"] == 2 and result["failed"] == 0
    # exact payloads: archive → is_archived only, nothing else
    assert all(p == {"is_archived": True} for _, p in calls), calls
    assert all(c[0].endswith("/api/bookmarks/101/") or c[0].endswith("/api/bookmarks/102/")
               for c in calls)


def test_apply_update_payloads_and_description_rule():
    ix = _ix()
    calls = []

    def fake_patch(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        return _FakeResp(200)

    ix._session.patch = fake_patch
    plan = ix.get_cleanup_plan(scope="redirects")
    result = ix.apply_cleanup(plan, scope="redirects")
    assert result["applied"] == 3 and result["failed"] == 0
    by_id = {p["bm_id"]: p for p in plan["planned"]}
    sent = {u.rsplit("/", 2)[-2]: j for u, j in calls}
    # 104: no description → URL + provenance note
    note = f"link-kb: updated from {by_id[104]['original']}"
    assert sent["104"] == {"url": "https://b.example/new", "description": note}, sent["104"]
    # 109: has description → URL only, note NOT clobbered
    assert sent["109"] == {"url": "https://b.example/desc"}, sent["109"]
    # 105: tracking params stripped in the written URL
    assert sent["105"]["url"] == "https://b.example/new?keep=1", sent["105"]


def test_apply_per_item_error_isolation():
    ix = _ix()
    calls = {"n": 0}

    def fake_patch(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResp(500, "boom")
        return _FakeResp(200)

    ix._session.patch = fake_patch
    plan = ix.get_cleanup_plan(scope="archive")
    result = ix.apply_cleanup(plan, scope="archive")
    assert result["applied"] == 1 and result["failed"] == 1
    assert any(not r["ok"] and "boom" in r["error"] for r in result["results"])


# --- /api/check-links route --------------------------------------------------------

class _StubSession:
    """Drops in for requests.Session — records every PATCH, always 200."""
    def __init__(self):
        self.patches = []

    def patch(self, url, headers=None, json=None, timeout=None):
        self.patches.append((url, json))
        return _FakeResp(200)


def _real_ix_with_stub():
    ix = Indexer()
    ix._health = dict(HEALTH)
    ix._fetch_linkding_links = lambda: [v for k, v in BM.items()]
    ix._session = _StubSession()
    return ix


def _client(ix):
    server.get_indexer = lambda: ix
    return server.app.test_client()


def test_route_get_dry_run():
    c = _client(_real_ix_with_stub())
    r = c.get("/api/check-links")
    assert r.status_code == 200
    body = r.get_json()
    assert body["dry_run"] is True
    assert body["counts"] == {"archive": 2, "update_url": 3}
    assert "applied" not in body  # nothing was written


def test_route_post_refused_when_gate_off(monkeypatch):
    monkeypatch.setattr(server, "AUTO_ARCHIVE", False)
    ix = _real_ix_with_stub()
    c = _client(ix)
    r = c.post("/api/check-links", json={"scope": "archive"})
    assert r.status_code == 403
    body = r.get_json()
    assert "refused" in body and body["dry_run"] is True
    assert "applied" not in body
    assert ix._session.patches == []  # nothing touched Linkding


def test_route_post_applies_when_gate_on(monkeypatch):
    monkeypatch.setattr(server, "AUTO_ARCHIVE", True)
    ix = _real_ix_with_stub()
    c = _client(ix)
    r = c.post("/api/check-links", json={"scope": "archive"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["applied"] == 2 and body["failed"] == 0
    assert body["dry_run"] is False
    assert all(j == {"is_archived": True} for _, j in ix._session.patches)
    assert len(ix._session.patches) == 2


def test_route_bad_scope_400():
    c = _client(_real_ix_with_stub())
    # GET ignores the scope query param (it's a read) and succeeds
    assert c.get("/api/check-links?scope=bogus").status_code == 200
    # POST with an unknown scope is a client error
    assert c.post("/api/check-links", json={"scope": "bogus"}).status_code == 400
