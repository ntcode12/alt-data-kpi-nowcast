import httpx
import pytest

from nowcast.http import CachedClient, OfflineCacheMiss, RateLimiter, make_user_agent
from nowcast.settings import Settings, SettingsError, load_settings


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _client(tmp_path, handler, **kw):
    kw.setdefault("max_per_sec", 8)
    return CachedClient(
        user_agent=make_user_agent("researcher@test.org"),
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        **kw,
    )


def test_user_agent_contains_contact_email(tmp_path):
    seen = {}

    def handler(req):
        seen["ua"] = req.headers["User-Agent"]
        return httpx.Response(200, json={"ok": 1})

    _client(tmp_path, handler).get_json("https://data.sec.gov/x", "sec/x.json")
    assert "researcher@test.org" in seen["ua"]
    assert seen["ua"].startswith("alt-data-kpi-nowcast/")


@pytest.mark.parametrize("email", [None, "", "you@example.com", "not-an-email"])
def test_settings_refuse_missing_or_placeholder_email(monkeypatch, tmp_path, email):
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    if email is not None:
        monkeypatch.setenv("CONTACT_EMAIL", email)
    with pytest.raises(SettingsError):
        load_settings(env_file=tmp_path / "missing.env").require_email()


def test_settings_read_email_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTACT_EMAIL", "researcher@test.org")
    s = load_settings(env_file=tmp_path / "missing.env")
    assert isinstance(s, Settings)
    assert s.require_email() == "researcher@test.org"


def test_rate_limiter_never_exceeds_rate():
    clock = FakeClock()
    rl = RateLimiter(8, clock=clock.now, sleep=clock.sleep)
    stamps = []
    for _ in range(20):
        rl.wait()
        stamps.append(clock.t)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(gaps) >= 1 / 8 - 1e-9
    # any 1-second window holds at most 10 requests (SEC fair-access limit)
    for i, t in enumerate(stamps):
        assert sum(1 for u in stamps[i:] if u < t + 1.0) <= 10


def test_rate_limiter_rejects_rates_above_sec_limit():
    with pytest.raises(ValueError):
        RateLimiter(11)


def test_client_requests_are_rate_limited(tmp_path):
    clock = FakeClock()
    stamps = []

    def handler(req):
        stamps.append(clock.t)
        return httpx.Response(200, json={})

    c = _client(tmp_path, handler, clock=clock.now, sleep=clock.sleep)
    for i in range(5):
        c.get_json(f"https://data.sec.gov/{i}", f"sec/{i}.json")
    assert all(b - a >= 1 / 8 - 1e-9 for a, b in zip(stamps, stamps[1:]))


def test_cache_hit_makes_no_request(tmp_path):
    calls = []

    def handler(req):
        calls.append(req.url)
        return httpx.Response(200, json={"v": 42})

    c = _client(tmp_path, handler)
    assert c.get_json("https://data.sec.gov/a", "sec/a.json") == {"v": 42}
    assert c.get_json("https://data.sec.gov/a", "sec/a.json") == {"v": 42}
    assert len(calls) == 1
    # a fresh client on the same cache dir also makes no request
    c2 = _client(tmp_path, handler)
    assert c2.get_json("https://data.sec.gov/a", "sec/a.json") == {"v": 42}
    assert len(calls) == 1


def test_offline_cache_miss_raises(tmp_path):
    def handler(req):  # pragma: no cover - must not be called
        raise AssertionError("network used in offline mode")

    with pytest.raises(OfflineCacheMiss):
        _client(tmp_path, handler, offline=True).get_json("https://x/y", "sec/y.json")


def test_retries_429_then_succeeds(tmp_path):
    clock = FakeClock()
    responses = iter([httpx.Response(429), httpx.Response(503), httpx.Response(200, json={"ok": True})])
    c = _client(tmp_path, lambda req: next(responses), clock=clock.now, sleep=clock.sleep)
    assert c.get_json("https://data.sec.gov/r", "sec/r.json") == {"ok": True}


def test_gives_up_after_retries_and_caches_nothing(tmp_path):
    clock = FakeClock()
    c = _client(tmp_path, lambda req: httpx.Response(500), clock=clock.now, sleep=clock.sleep)
    with pytest.raises(httpx.HTTPStatusError):
        c.get_json("https://data.sec.gov/r", "sec/r.json")
    assert not (tmp_path / "sec/r.json").exists()


def test_404_is_not_retried(tmp_path):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        _client(tmp_path, handler).get_json("https://x/404", "w/404.json")
    assert len(calls) == 1


def test_get_text_cached_and_offline(tmp_path):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(200, text="<html>hi</html>")

    c = _client(tmp_path, handler)
    assert c.get_text("https://www.sec.gov/a", "sec/a.html") == "<html>hi</html>"
    assert c.get_text("https://www.sec.gov/a", "sec/a.html") == "<html>hi</html>"
    assert len(calls) == 1
    with pytest.raises(OfflineCacheMiss):
        _client(tmp_path, handler, offline=True).get_text("https://x/b", "sec/b.html")
