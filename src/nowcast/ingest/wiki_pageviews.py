"""Wikimedia REST pageviews (en.wikipedia, all-access, agent=user) -> daily views per article."""

from __future__ import annotations

import datetime as dt
from urllib.parse import quote

import pandas as pd

BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
PAGEVIEWS_EPOCH = dt.date(2015, 7, 1)  # earliest data in the REST API


def encode_title(article: str) -> str:
    return quote(article.replace(" ", "_"), safe="")


def article_url(article: str, start: dt.date, end: dt.date) -> str:
    return f"{BASE}/{encode_title(article)}/daily/{start:%Y%m%d}/{end:%Y%m%d}"


def fetch_pageviews(article: str, start: dt.date, end: dt.date, client) -> dict:
    key = f"wiki/pageviews/{encode_title(article)}__{start:%Y%m%d}_{end:%Y%m%d}.json"
    return client.get_json(article_url(article, start, end), key)


def parse_pageviews(doc: dict, article: str) -> pd.DataFrame:
    """Days missing from the API are left missing (coverage is measured later, never imputed)."""
    rows = [
        {
            "article": article,
            "date": dt.datetime.strptime(it["timestamp"][:8], "%Y%m%d").date(),
            "views": int(it["views"]),
        }
        for it in doc.get("items", [])
    ]
    return pd.DataFrame(rows, columns=["article", "date", "views"])


SITE_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/en.wikipedia/all-access/user/daily"


def site_url(start: dt.date, end: dt.date) -> str:
    return f"{SITE_BASE}/{start:%Y%m%d}00/{end:%Y%m%d}00"


def fetch_site_pageviews(start: dt.date, end: dt.date, client) -> dict:
    """Total en.wikipedia human pageviews per day (used to normalize for site-wide traffic shifts)."""
    return client.get_json(site_url(start, end), f"wiki/site/en.wikipedia__{start:%Y%m%d}_{end:%Y%m%d}.json")


def parse_site_pageviews(doc: dict) -> pd.DataFrame:
    df = parse_pageviews(doc, "")
    return df[["date", "views"]]
