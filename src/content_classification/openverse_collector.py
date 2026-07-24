import argparse
import json
import re
import urllib.error
from pathlib import Path

from src.data_generation.image_downloader import descargar_imagenes_openverse


QUERIES = {
    "animales": [
        "dog animal",
        "cat animal",
        "horse animal",
        "bird wildlife",
        "elephant animal",
        "cow animal",
        "sheep animal",
        "zebra wildlife",
        "giraffe wildlife",
        "bear wildlife",
    ],
    "frutas": [
        "apple fruit",
        "banana fruit",
        "orange fruit",
        "strawberry fruit",
        "grapes fruit",
        "pineapple fruit",
        "watermelon fruit",
        "pear fruit",
        "peach fruit",
        "mango fruit",
    ],
    "objetos": [
        "chair furniture",
        "coffee mug",
        "smartphone device",
        "laptop isolated",
        "table lamp",
        "wristwatch object",
        "backpack object",
        "bicycle object",
        "kitchen bottle",
        "television object",
    ],
    "personas": [
        "person portrait",
        "people group",
        "woman portrait",
        "man portrait",
        "child portrait",
        "student person",
        "worker portrait",
        "athlete person",
        "elderly person",
        "family people",
    ],
}


def safe_name(text):
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.lower()).strip("_")
    return name


def collect_openverse(output_dir, images_per_query):
    if images_per_query < 1:
        raise ValueError("La cantidad de imágenes debe ser mayor que cero")

    output_dir = Path(output_dir)
    records = []
    errors = []
    counts = {}

    for class_name, queries in QUERIES.items():
        counts[class_name] = 0
        for query in queries:
            query_dir = output_dir / class_name / safe_name(query)
            try:
                downloaded = descargar_imagenes_openverse(
                    query,
                    images_per_query,
                    query_dir,
                )
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                errors.append(
                    {
                        "clase": class_name,
                        "busqueda": query,
                        "error": str(error),
                    }
                )
                continue

            for image in downloaded:
                record = dict(image)
                record["clase"] = class_name
                record["busqueda"] = query
                record["archivo"] = str(query_dir / image["archivo"])
                records.append(record)
                counts[class_name] += 1

            print(
                f"{class_name} - {query}: "
                f"{len(downloaded)} imágenes"
            )

    manifest = {
        "fuente": "Openverse",
        "cantidad": len(records),
        "imagenes_por_busqueda": images_per_query,
        "cantidad_por_clase": counts,
        "imagenes": records,
        "errores": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "openverse_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sources/openverse"),
    )
    parser.add_argument("--images-per-query", type=int, default=50)
    args = parser.parse_args()

    manifest = collect_openverse(args.output_dir, args.images_per_query)
    print(f"Imágenes descargadas: {manifest['cantidad']}")
    print(f"Búsquedas con error: {len(manifest['errores'])}")
    print(f"Manifiesto: {args.output_dir / 'openverse_manifest.json'}")


if __name__ == "__main__":
    main()
