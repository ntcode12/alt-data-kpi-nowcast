"""Resolve Wikipedia titles to their canonical page (the pageviews API does not follow redirects)."""

from __future__ import annotations

from urllib.parse import urlencode

from nowcast.ingest.wiki_pageviews import encode_title

API = "https://en.wikipedia.org/w/api.php"


def resolve_url(article: str) -> str:
    return API + "?" + urlencode(
        {"action": "query", "titles": article, "redirects": 1, "format": "json", "formatversion": 2}
    )


def resolve_title(article: str, client) -> tuple[str, bool]:
    """Return (canonical_title, exists)."""
    doc = client.get_json(resolve_url(article), f"wiki/titles/{encode_title(article)}.json")
    q = doc.get("query", {})
    title = article
    for step in q.get("normalized", []) + q.get("redirects", []):
        if step["from"] == title:
            title = step["to"]
    pages = q.get("pages", [])
    exists = bool(pages) and not pages[0].get("missing", False)
    return title, exists
