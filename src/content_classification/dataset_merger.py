import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2

from src.content_classification.coco_dataset import calculate_split_sizes
from src.content_classification.dataset import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    SPLIT_NAMES,
)


def find_images(source_dir, class_name):
    source_dir = Path(source_dir)
    directories = []

    direct_class_dir = source_dir / class_name
    if direct_class_dir.is_dir():
        directories.append(direct_class_dir)

    for split_name in SPLIT_NAMES:
        split_class_dir = source_dir / split_name / class_name
        if split_class_dir.is_dir():
            directories.append(split_class_dir)

    images = []
    for directory in directories:
        images.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(images)


def image_hash(path):
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None

    resized = cv2.resize(image, (9, 8))
    differences = resized[:, 1:] > resized[:, :-1]
    visual_hash = 0
    for different in differences.flatten():
        visual_hash = (visual_hash << 1) | int(different)
    return content_hash, visual_hash


def is_visual_duplicate(visual_hash, known_hashes):
    for known_hash in known_hashes:
        if (visual_hash ^ known_hash).bit_count() <= 3:
            return True
    return False


def collect_unique_images(source_dirs):
    selected = {class_name: [] for class_name in CLASS_NAMES}
    content_hashes = set()
    visual_hashes = []
    invalid = []
    duplicates = []

    for source_dir in source_dirs:
        for class_name in CLASS_NAMES:
            for path in find_images(source_dir, class_name):
                hashes = image_hash(path)
                if hashes is None:
                    invalid.append(str(path))
                    continue

                content_hash, visual_hash = hashes
                if (
                    content_hash in content_hashes
                    or is_visual_duplicate(visual_hash, visual_hashes)
                ):
                    duplicates.append(str(path))
                    continue

                content_hashes.add(content_hash)
                visual_hashes.append(visual_hash)
                selected[class_name].append(
                    {
                        "path": path,
                        "source": str(Path(source_dir)),
                        "hash": content_hash,
                    }
                )

    return selected, invalid, duplicates


def merge_datasets(source_dirs, output_dir, images_per_class, seed):
    if not source_dirs:
        raise ValueError("Debes indicar al menos una fuente")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"La carpeta de salida no está vacía: {output_dir}")

    selected, invalid, duplicates = collect_unique_images(source_dirs)
    random_generator = random.Random(seed)

    for class_name in CLASS_NAMES:
        random_generator.shuffle(selected[class_name])
        available = len(selected[class_name])
        if available < images_per_class:
            raise ValueError(
                f"No hay suficientes imágenes para {class_name}: "
                f"se necesitan {images_per_class} y hay {available}"
            )

    split_sizes = calculate_split_sizes(images_per_class)
    manifest = {
        "sources": [str(Path(source)) for source in source_dirs],
        "images_per_class": images_per_class,
        "seed": seed,
        "duplicates_removed": len(duplicates),
        "invalid_removed": len(invalid),
        "splits": {},
    }

    positions = {class_name: 0 for class_name in CLASS_NAMES}
    for split_name, split_size in split_sizes.items():
        manifest["splits"][split_name] = {}

        for class_name in CLASS_NAMES:
            start = positions[class_name]
            end = start + split_size
            images = selected[class_name][start:end]
            positions[class_name] = end
            class_dir = output_dir / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            records = []

            for number, image in enumerate(images, start=1):
                suffix = image["path"].suffix.lower()
                if suffix == ".jpeg":
                    suffix = ".jpg"
                filename = f"{class_name}_{number:04d}_{image['hash'][:10]}{suffix}"
                destination = class_dir / filename
                shutil.copy2(image["path"], destination)
                records.append(
                    {
                        "file": str(destination),
                        "original": str(image["path"]),
                        "source": image["source"],
                        "hash": image["hash"],
                    }
                )

            manifest["splits"][split_name][class_name] = records

    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        manifest = merge_datasets(
            args.source,
            args.output_dir,
            args.images_per_class,
            args.seed,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    print(f"Dataset creado en: {args.output_dir}")
    print(f"Imágenes por clase: {manifest['images_per_class']}")
    print(f"Duplicados eliminados: {manifest['duplicates_removed']}")
    print(f"Imágenes inválidas: {manifest['invalid_removed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
