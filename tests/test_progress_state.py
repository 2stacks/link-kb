"""v1.0.18 regression: in-memory progress state across consecutive runs.

Pre-fix, trigger_full_index() reassigned _full_progress WITHOUT a `global`
declaration, so the reset created a throwaway local dict and a completed
run's done=True leaked into the next run. The UI gate
(`full_indexing && !done`) then hid the progress bar for every run after
the first. The same missing declaration made the already_running 409
branch read an unbound local (500) when the button was clicked mid-run.
"""
import time

from app import server


class _FakeIndexer:
    def full_index(self, progress_callback=None):
        for i in range(1, 5):
            time.sleep(0.15)
            if progress_callback:
                progress_callback(i, 4, f"https://a.example/{i}")
        return 4

    def diff_index(self, progress_callback=None):
        for i in range(1, 3):
            time.sleep(0.15)
            if progress_callback:
                progress_callback(i, 2, f"https://a.example/{i}")
        return {"added": 2, "removed": 0, "unchanged": 1}

    def get_status(self):
        return {"total_indexed": 4, "linkding_count": 4}


_fake = _FakeIndexer()


def _client():
    server.get_indexer = lambda: _fake
    return server.app.test_client()


def _reset_server_state():
    server._full_indexing = False
    server._diff_indexing = False
    server._full_progress = {"total": 0, "processed": 0, "started_at": None, "done": False}
    server._diff_progress = {"total": 0, "processed": 0, "started_at": None, "done": False}


def _wait(cond, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.1)
    return False


def _status(client):
    return client.get("/api/status").get_json()


def test_second_run_shows_progress():
    _reset_server_state()
    c = _client()

    # Run 1 to completion
    assert c.post("/api/full-index").status_code == 200
    assert _wait(lambda: _status(c)["full_progress"]["done"])
    assert _status(c)["full_progress"]["done"] is True
    assert _status(c)["full_indexing"] is False

    # Run 2 — the bug: done=True from run 1 hid all progress
    assert c.post("/api/full-index").status_code == 200
    ok = _wait(lambda: (
        _status(c)["full_indexing"]
        and not _status(c)["full_progress"]["done"]
        and _status(c)["full_progress"]["processed"] > 0
    ))
    assert ok, f"run 2 in-flight state never visible: {_status(c)}"
    assert _wait(lambda: _status(c)["full_progress"]["done"])


def test_double_click_returns_409_with_progress():
    _reset_server_state()
    c = _client()

    assert c.post("/api/full-index").status_code == 200
    assert _wait(lambda: _status(c)["full_indexing"])
    try:
        resp = c.post("/api/full-index")
        body = resp.get_json(silent=True)
        assert resp.status_code == 409, f"expected 409, got {resp.status_code}: {resp.data[:120]}"
        assert body and body.get("status") == "already_running"
        assert "progress" in body
    finally:
        assert _wait(lambda: _status(c)["full_progress"]["done"])
