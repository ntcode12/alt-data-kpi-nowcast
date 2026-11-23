import datetime as dt

import pandas as pd

from nowcast.ingest.sec_filings import (
    assign_period_end,
    earnings_8ks,
    exhibit_filename,
    html_to_text,
    submissions_url,
)

D = dt.date


def _subs(forms, items, dates, accns, docs):
    return {"filings": {"recent": {"form": forms, "items": items, "filingDate": dates,
                                   "accessionNumber": accns, "primaryDocument": docs}, "files": []}}


def test_submissions_url_zero_pads():
    assert submissions_url(1065280) == "https://data.sec.gov/submissions/CIK0001065280.json"


def test_earnings_8ks_filters_item_202():
    doc = _subs(["8-K", "8-K", "10-Q", "8-K/A"], ["2.02,9.01", "5.02", "", "2.02"],
                ["2024-01-23", "2024-02-01", "2024-04-20", "2024-01-25"],
                ["a-1", "a-2", "a-3", "a-4"], ["x.htm"] * 4)
    df = earnings_8ks([doc])
    assert df["accn"].tolist() == ["a-1"]          # amendments excluded; first filing is the as-reported one
    assert df["filed"].tolist() == [D(2024, 1, 23)]


def test_earnings_8ks_merges_older_pages_and_dedups():
    a = _subs(["8-K"], ["2.02"], ["2024-01-23"], ["a-1"], ["x.htm"])
    b = {"form": ["8-K", "8-K"], "items": ["2.02", "2.02"], "filingDate": ["2016-01-19", "2024-01-23"],
         "accessionNumber": ["b-1", "a-1"], "primaryDocument": ["y.htm"] * 2}
    df = earnings_8ks([a, b])
    assert df["accn"].tolist() == ["b-1", "a-1"]


def test_exhibit_filename_from_index_headers():
    hdr = """<DOCUMENT>
<TYPE>8-K
<FILENAME>nflx-20240123.htm
</DOCUMENT><DOCUMENT>
<TYPE>EX-99.1
<FILENAME>ex991_q423.htm
<DESCRIPTION>EX-99.1
</DOCUMENT>"""
    assert exhibit_filename(hdr) == "ex991_q423.htm"
    assert exhibit_filename("<TYPE>EX-99\n<FILENAME>pr.htm") == "pr.htm"
    assert exhibit_filename("<TYPE>8-K\n<FILENAME>a.htm") is None


def test_html_to_text_keeps_table_cells_apart():
    html = "<p>Revenue&nbsp;grew</p><table><tr><td>Revenue</td><td>$ 9,370,440</td></tr></table><script>x</script>"
    t = html_to_text(html)
    assert "Revenue grew" in t
    assert "Revenue | $ 9,370,440" in t
    assert "x" not in t.split()


def test_assign_period_end_is_latest_quarter_ended_before_release():
    q = pd.DataFrame({"ticker": ["T"] * 3, "end": [D(2023, 9, 30), D(2023, 12, 31), D(2024, 3, 31)]})
    rel = pd.DataFrame({"ticker": ["T", "T"], "filed": [D(2024, 1, 23), D(2023, 12, 1)]})
    out = assign_period_end(rel, q)
    assert out["period_end"].tolist() == [D(2023, 12, 31), D(2023, 9, 30)]


def test_assign_period_end_respects_window():
    q = pd.DataFrame({"ticker": ["T"], "end": [D(2023, 9, 30)]})
    rel = pd.DataFrame({"ticker": ["T", "T", "T"], "filed": [D(2023, 10, 20), D(2024, 3, 1), D(2023, 9, 30)]})
    out = assign_period_end(rel, q)
    assert out["period_end"].tolist() == [D(2023, 9, 30), None, None]   # >75 days; same-day is not "after"


def test_exhibit_filename_real_edgar_format_escaped_with_sequence():
    hdr = """<PRE>&lt;DOCUMENT&gt;
&lt;TYPE&gt;8-K
&lt;SEQUENCE&gt;1
&lt;FILENAME&gt;cmg-20150421x8k.htm
&lt;/DOCUMENT&gt;
&lt;DOCUMENT&gt;
&lt;TYPE&gt;EX-99.1
&lt;SEQUENCE&gt;2
&lt;FILENAME&gt;cmg-20150421ex9911991eb.htm
&lt;DESCRIPTION&gt;EX-99.1
&lt;/DOCUMENT&gt;</PRE>"""
    assert exhibit_filename(hdr) == "cmg-20150421ex9911991eb.htm"
