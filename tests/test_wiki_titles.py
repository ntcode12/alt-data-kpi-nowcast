from nowcast.ingest.wiki_titles import resolve_title, resolve_url


class Stub:
    def __init__(self, doc):
        self.doc = doc

    def get_json(self, url, cache_key):
        self.url, self.key = url, cache_key
        return self.doc


def test_follows_normalization_and_redirect():
    doc = {"query": {"normalized": [{"from": "lululemon", "to": "Lululemon"}],
                     "redirects": [{"from": "Lululemon", "to": "Lululemon Athletica"}],
                     "pages": [{"title": "Lululemon Athletica", "pageid": 1}]}}
    assert resolve_title("lululemon", Stub(doc)) == ("Lululemon Athletica", True)


def test_missing_page():
    doc = {"query": {"pages": [{"title": "Nope", "missing": True}]}}
    assert resolve_title("Nope", Stub(doc)) == ("Nope", False)


def test_url_requests_redirect_resolution():
    assert "redirects=1" in resolve_url("Zoom (software)")
