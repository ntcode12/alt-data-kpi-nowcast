"""8-K Item 2.02 earnings releases (exhibit EX-99.1) -> raw_releases."""

from __future__ import annotations

import datetime as dt
import html as htmllib
import re

import pandas as pd

RELEASE_WINDOW_DAYS = 75   # releases come 15-60 days after quarter end


def submissions_url(cik: int) -> str:
    return f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"


def fetch_submissions(cik: int, client) -> list[dict]:
    """Main submissions doc plus any older pages listed under filings.files."""
    main = client.get_json(submissions_url(cik), f"sec/submissions/CIK{int(cik):010d}.json")
    pages = [main]
    for f in main.get("filings", {}).get("files", []):
        pages.append(client.get_json(f"https://data.sec.gov/submissions/{f['name']}", f"sec/submissions/{f['name']}"))
    return pages


def earnings_8ks(pages: list[dict]) -> pd.DataFrame:
    rows = []
    for page in pages:
        r = page["filings"]["recent"] if "filings" in page else page
        for form, items, filed, accn, doc in zip(r["form"], r["items"], r["filingDate"],
                                                 r["accessionNumber"], r["primaryDocument"]):
            if form == "8-K" and "2.02" in (items or "").split(","):
                rows.append(dict(accn=accn, filed=dt.date.fromisoformat(filed), primary_doc=doc))
    df = pd.DataFrame(rows, columns=["accn", "filed", "primary_doc"]).drop_duplicates("accn")
    return df.sort_values("filed").reset_index(drop=True)


def index_headers_url(cik: int, accn: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}/{accn}-index-headers.html"


def exhibit_filename(headers: str) -> str | None:
    """First document whose TYPE is EX-99.1 (or EX-99 when unnumbered)."""
    text = htmllib.unescape(headers)
    pairs = []
    for block in re.split(r"<DOCUMENT>", text)[1:] or [text]:
        t = re.search(r"<TYPE>([^\n<]+)", block)
        f = re.search(r"<FILENAME>([^\n<]+)", block)
        if t and f:
            pairs.append((t.group(1).strip().upper(), f.group(1).strip()))
    for want in ("EX-99.1", "EX-99.01", "EX-99"):
        for t, f in pairs:
            if t == want:
                return f
    return None


def html_to_text(raw: str) -> str:
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    s = re.sub(r"(?i)</t[dh]>", " | ", s)
    s = re.sub(r"(?i)<(br|/p|/div|/tr|/li|/h\d)[^>]*>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = htmllib.unescape(s).replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"( \| )+\n", "\n", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def fetch_release_text(cik: int, accn: str, client) -> tuple[str | None, str | None]:
    """Returns (exhibit filename, cleaned text). Uses a text cache since EDGAR serves HTML, not JSON."""
    hdr = client.get_text(index_headers_url(cik, accn), f"sec/releases/{accn}-index-headers.html")
    fname = exhibit_filename(hdr)
    if fname is None:
        return None, None
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}/{fname}"
    return fname, html_to_text(client.get_text(url, f"sec/releases/{accn}-{fname}"))


def assign_period_end(releases: pd.DataFrame, quarters: pd.DataFrame) -> pd.DataFrame:
    """Fiscal quarter a release reports on: the latest quarter ending strictly before the release, within 75 days."""
    out = []
    for r in releases.itertuples():
        ends = quarters.loc[(quarters["ticker"] == r.ticker) & (quarters["end"] < r.filed), "end"]
        e = ends.max() if len(ends) else None
        out.append(e if e is not None and (r.filed - e).days <= RELEASE_WINDOW_DAYS else None)
    return releases.assign(period_end=out)
