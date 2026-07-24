import json

import src.content_classification.openverse_collector as collector


def test_safe_name():
    assert collector.safe_name("Coffee mug") == "coffee_mug"


def test_collect_openverse(tmp_path, monkeypatch):
    calls = []

    def fake_download(query, limit, output_dir):
        calls.append((query, limit, output_dir))
        output_dir.mkdir(parents=True)
        image_path = output_dir / "001_image.jpg"
        image_path.write_bytes(b"image")
        return [
            {
                "archivo": image_path.name,
                "titulo": query,
                "autor": "Autor",
                "licencia": "cc0",
                "licencia_url": "https://example.com/license",
                "fuente": "https://example.com/source",
                "url_descarga": "https://example.com/image.jpg",
            }
        ]

    monkeypatch.setattr(
        collector,
        "descargar_imagenes_openverse",
        fake_download,
    )

    output_dir = tmp_path / "openverse"
    manifest = collector.collect_openverse(output_dir, 4)

    assert len(calls) == 40
    assert manifest["cantidad"] == 40
    assert manifest["cantidad_por_clase"] == {
        "animales": 10,
        "frutas": 10,
        "objetos": 10,
        "personas": 10,
    }
    assert {image["clase"] for image in manifest["imagenes"]} == {
        "animales",
        "frutas",
        "objetos",
        "personas",
    }

    saved = json.loads(
        (output_dir / "openverse_manifest.json").read_text(encoding="utf-8")
    )
    assert saved["imagenes"][0]["archivo"].endswith("001_image.jpg")


def test_collect_openverse_rejects_zero(tmp_path):
    try:
        collector.collect_openverse(tmp_path, 0)
    except ValueError as error:
        assert "mayor que cero" in str(error)
    else:
        raise AssertionError("Debía rechazar una cantidad igual a cero")


def test_collect_openverse_continues_after_error(tmp_path, monkeypatch):
    def fake_download(query, limit, output_dir):
        raise TimeoutError("Tiempo agotado")

    monkeypatch.setattr(
        collector,
        "descargar_imagenes_openverse",
        fake_download,
    )

    manifest = collector.collect_openverse(tmp_path, 2)

    assert manifest["cantidad"] == 0
    assert len(manifest["errores"]) == 40
