import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import cv2

OPENVERSE_ENDPOINT = "https://api.openverse.org/v1/images/"
USER_AGENT = "TIF-grafica-data-generator/1.0"


def _abrir_url(url: str, timeout: int = 30) -> bytes:
    solicitud = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(solicitud, timeout=timeout) as respuesta:
        return respuesta.read()


def _nombre_seguro(texto: str) -> str:
    nombre = re.sub(r"[^a-zA-Z0-9_-]+", "_", texto.strip().lower()).strip("_")
    return nombre or "imagen"


def buscar_openverse(query: str, limite: int) -> list[dict[str, Any]]:
    resultados = []
    pagina = 1
    cantidad_buscada = max(limite * 2, limite)

    while len(resultados) < cantidad_buscada:
        cantidad_pagina = min(20, cantidad_buscada - len(resultados))
        parametros = {
            "q": query,
            "page": pagina,
            "page_size": cantidad_pagina,
            "extension": "jpg,png,jpeg",
            "license_type": "commercial,modification",
        }
        url = f"{OPENVERSE_ENDPOINT}?{urllib.parse.urlencode(parametros)}"
        datos = json.loads(_abrir_url(url).decode("utf-8"))
        imagenes = datos.get("results", [])
        if not imagenes:
            break
        resultados.extend(imagenes)
        if not datos.get("next"):
            break
        pagina += 1

    return resultados


def _extension_desde_url(url: str) -> str:
    ruta = urllib.parse.urlparse(url).path.lower()
    for extension in (".jpg", ".jpeg", ".png"):
        if ruta.endswith(extension):
            return ".jpg" if extension == ".jpeg" else extension
    return ".jpg"


def _es_imagen_valida(ruta: Path) -> bool:
    imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    return imagen is not None and imagen.size > 0


def descargar_imagenes_openverse(query: str, limite: int, salida: Path) -> list[dict[str, Any]]:
    salida.mkdir(parents=True, exist_ok=True)
    resultados = buscar_openverse(query, limite)
    descargadas: list[dict[str, Any]] = []

    for item in resultados:
        if len(descargadas) >= limite:
            break

        url_imagen = item.get("url")
        if not url_imagen:
            continue

        nombre_base = _nombre_seguro(item.get("title") or f"{query}_{len(descargadas) + 1}")
        extension = _extension_desde_url(url_imagen)
        ruta = salida / f"{len(descargadas) + 1:03d}_{nombre_base[:60]}{extension}"

        try:
            ruta.write_bytes(_abrir_url(url_imagen))
        except (OSError, urllib.error.URLError, TimeoutError):
            continue

        if not _es_imagen_valida(ruta):
            ruta.unlink(missing_ok=True)
            continue

        descargadas.append(
            {
                "archivo": ruta.name,
                "titulo": item.get("title"),
                "autor": item.get("creator"),
                "licencia": item.get("license"),
                "licencia_url": item.get("license_url"),
                "fuente": item.get("foreign_landing_url"),
                "url_descarga": url_imagen,
            }
        )

    manifiesto = salida / "openverse_manifest.json"
    manifiesto.write_text(
        json.dumps(
            {"query": query, "cantidad": len(descargadas), "imagenes": descargadas},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return descargadas
