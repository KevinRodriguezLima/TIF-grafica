import json
import urllib.parse

import src.data_generation.image_downloader as downloader


def test_searches_openverse_with_small_pages(monkeypatch):
    pages = []

    def fake_open(url, timeout=30):
        parameters = urllib.parse.parse_qs(
            urllib.parse.urlparse(url).query
        )
        page = int(parameters["page"][0])
        page_size = int(parameters["page_size"][0])
        pages.append((page, page_size))
        result = {
            "results": [{"id": number} for number in range(page_size)],
            "next": "next" if page < 3 else None,
        }
        return json.dumps(result).encode("utf-8")

    monkeypatch.setattr(downloader, "_abrir_url", fake_open)

    results = downloader.buscar_openverse("dog animal", 25)

    assert len(results) == 50
    assert pages == [(1, 20), (2, 20), (3, 10)]
