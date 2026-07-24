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


def load_existing(query_dir, images_per_query):
    manifest_path = query_dir / "openverse_manifest.json"
    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = manifest.get("imagenes", [])
    valid_images = [
        image
        for image in images
        if (query_dir / image["archivo"]).is_file()
    ]
    if len(valid_images) < images_per_query:
        return None
    return valid_images[:images_per_query]


def save_manifest(output_dir, images_per_query, records, errors, counts):
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


def collect_openverse(output_dir, images_per_query, class_name=None):
    if images_per_query < 1:
        raise ValueError("La cantidad de imágenes debe ser mayor que cero")
    if class_name is not None and class_name not in QUERIES:
        raise ValueError(f"Clase no válida: {class_name}")

    output_dir = Path(output_dir)
    records = []
    errors = []
    counts = {}
    classes = [class_name] if class_name else QUERIES
    general_manifest = output_dir / "openverse_manifest.json"

    if class_name and general_manifest.is_file():
        previous = json.loads(general_manifest.read_text(encoding="utf-8"))
        records = [
            image
            for image in previous.get("imagenes", [])
            if image.get("clase") != class_name
        ]
        errors = [
            error
            for error in previous.get("errores", [])
            if error.get("clase") != class_name
        ]
        counts = {
            name: count
            for name, count in previous.get("cantidad_por_clase", {}).items()
            if name != class_name
        }

    for current_class in classes:
        counts[current_class] = 0
        queries = QUERIES[current_class]
        for query in queries:
            query_dir = output_dir / current_class / safe_name(query)
            downloaded = load_existing(query_dir, images_per_query)

            if downloaded is None:
                try:
                    downloaded = descargar_imagenes_openverse(
                        query,
                        images_per_query,
                        query_dir,
                    )
                except (OSError, urllib.error.URLError, TimeoutError) as error:
                    errors.append(
                        {
                            "clase": current_class,
                            "busqueda": query,
                            "error": str(error),
                        }
                    )
                    save_manifest(
                        output_dir,
                        images_per_query,
                        records,
                        errors,
                        counts,
                    )
                    continue

            for image in downloaded:
                record = dict(image)
                record["clase"] = current_class
                record["busqueda"] = query
                record["archivo"] = str(query_dir / image["archivo"])
                records.append(record)
                counts[current_class] += 1

            print(
                f"{current_class} - {query}: "
                f"{len(downloaded)} imágenes"
            )
            save_manifest(
                output_dir,
                images_per_query,
                records,
                errors,
                counts,
            )

    return save_manifest(
        output_dir,
        images_per_query,
        records,
        errors,
        counts,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sources/openverse"),
    )
    parser.add_argument("--images-per-query", type=int, default=50)
    parser.add_argument("--class-name", choices=QUERIES, default=None)
    args = parser.parse_args()

    manifest = collect_openverse(
        args.output_dir,
        args.images_per_query,
        args.class_name,
    )
    print(f"Imágenes descargadas: {manifest['cantidad']}")
    print(f"Búsquedas con error: {len(manifest['errores'])}")
    print(f"Manifiesto: {args.output_dir / 'openverse_manifest.json'}")


if __name__ == "__main__":
    main()
