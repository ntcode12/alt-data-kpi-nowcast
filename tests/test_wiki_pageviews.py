import datetime as dt
import json
from pathlib import Path

from nowcast.ingest.wiki_pageviews import article_url, fetch_pageviews, parse_pageviews

FIXTURE = Path(__file__).parent / "fixtures" / "pageviews_min.json"
BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"


def test_url_encodes_title_and_uses_user_agent_filter():
    url = article_url("Zoom (software)", dt.date(2015, 7, 1), dt.date(2026, 11, 22))
    assert url == BASE + "Zoom_%28software%29/daily/20150701/20261122"


def test_url_spaces_become_underscores():
    url = article_url("Chipotle Mexican Grill", dt.date(2020, 1, 1), dt.date(2020, 1, 31))
    assert url == BASE + "Chipotle_Mexican_Grill/daily/20200101/20200131"


def test_url_escapes_slashes_in_titles():
    assert "AC%2FDC" in article_url("AC/DC", dt.date(2020, 1, 1), dt.date(2020, 1, 2))


def test_parse_pageviews():
    df = parse_pageviews(json.loads(FIXTURE.read_text()), "Zoom (software)")
    assert list(df.columns) == ["article", "date", "views"]
    assert (df["article"] == "Zoom (software)").all()
    assert df["date"].tolist() == [dt.date(2020, 3, 1), dt.date(2020, 3, 2), dt.date(2020, 3, 4)]
    assert df["views"].tolist() == [1200, 1350, 98000]


def test_parse_does_not_invent_missing_days():
    df = parse_pageviews(json.loads(FIXTURE.read_text()), "Zoom (software)")
    assert dt.date(2020, 3, 3) not in set(df["date"])


def test_parse_empty():
    df = parse_pageviews({"items": []}, "X")
    assert df.empty and list(df.columns) == ["article", "date", "views"]


def test_fetch_cache_key_is_stable_and_filesystem_safe():
    class Spy:
        def get_json(self, url, cache_key):
            self.key = cache_key
            return {"items": []}

    spy = Spy()
    fetch_pageviews("Zoom (software)", dt.date(2015, 7, 1), dt.date(2026, 11, 22), spy)
    assert spy.key == "wiki/pageviews/Zoom_%28software%29__20150701_20261122.json"


def test_site_total_url_and_parse():
    from nowcast.ingest.wiki_pageviews import fetch_site_pageviews, parse_site_pageviews, site_url

    assert site_url(dt.date(2015, 7, 1), dt.date(2026, 11, 22)) == (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/en.wikipedia/all-access/user/daily/"
        "2015070100/2026112200"
    )
    doc = {"items": [{"project": "en.wikipedia", "timestamp": "2020010100", "views": 250000000}]}
    df = parse_site_pageviews(doc)
    assert list(df.columns) == ["date", "views"] and df["views"].tolist() == [250000000]

    class Spy:
        def get_json(self, url, cache_key):
            self.key = cache_key
            return doc

    spy = Spy()
    fetch_site_pageviews(dt.date(2015, 7, 1), dt.date(2026, 11, 22), spy)
    assert spy.key == "wiki/site/en.wikipedia__20150701_20261122.json"
