import argparse
import json
import random
import urllib.error
import urllib.request
from pathlib import Path

import cv2

from src.content_classification.dataset import CLASS_NAMES


ANIMAL_CATEGORIES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}
FRUIT_CATEGORIES = {"banana", "apple", "orange"}


def group_category(category_name):
    if category_name == "person":
        return "personas"
    if category_name in ANIMAL_CATEGORIES:
        return "animales"
    if category_name in FRUIT_CATEGORIES:
        return "frutas"
    return "objetos"


def load_candidates(annotations_path, min_dominance, min_image_area, seed):
    data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    categories = {
        category["id"]: group_category(category["name"])
        for category in data["categories"]
    }
    images = {image["id"]: image for image in data["images"]}
    areas = {}

    for annotation in data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        image_id = annotation["image_id"]
        class_name = categories[annotation["category_id"]]
        width = annotation["bbox"][2]
        height = annotation["bbox"][3]
        area = width * height
        if image_id not in areas:
            areas[image_id] = {}
        areas[image_id][class_name] = areas[image_id].get(class_name, 0) + area

    candidates = {class_name: [] for class_name in CLASS_NAMES}
    for image_id, class_areas in areas.items():
        total_area = sum(class_areas.values())
        class_name = max(class_areas, key=class_areas.get)
        dominant_area = class_areas[class_name]
        image = images[image_id]
        image_area = image["width"] * image["height"]

        if dominant_area / total_area < min_dominance:
            continue
        if dominant_area / image_area < min_image_area:
            continue

        candidates[class_name].append(image)

    random_generator = random.Random(seed)
    for images_list in candidates.values():
        random_generator.shuffle(images_list)
    return candidates


def calculate_split_sizes(images_per_class):
    if images_per_class < 3:
        raise ValueError("images_per_class debe ser al menos 3")

    validation_size = max(1, int(images_per_class * 0.15))
    test_size = max(1, int(images_per_class * 0.15))
    train_size = images_per_class - validation_size - test_size
    return {
        "train": train_size,
        "validation": validation_size,
        "test": test_size,
    }


def download_image(image, destination):
    url = image.get("coco_url") or image.get("url")
    if not url:
        return False

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "TIF-grafica-dataset/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            destination.write_bytes(response.read())
    except (OSError, urllib.error.URLError, TimeoutError):
        return False

    loaded_image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    if loaded_image is None or loaded_image.size == 0:
        destination.unlink(missing_ok=True)
        return False
    return True


def create_dataset(
    annotations_path,
    output_dir,
    images_per_class,
    min_dominance,
    min_image_area,
    seed,
):
    candidates = load_candidates(
        annotations_path,
        min_dominance,
        min_image_area,
        seed,
    )
    split_sizes = calculate_split_sizes(images_per_class)
    manifest = {
        "source": "COCO 2017",
        "annotations": str(Path(annotations_path).as_posix()),
        "images_per_class": images_per_class,
        "min_dominance": min_dominance,
        "min_image_area": min_image_area,
        "seed": seed,
        "splits": {},
    }

    for class_name in CLASS_NAMES:
        required = images_per_class
        available = len(candidates[class_name])
        if available < required:
            raise ValueError(
                f"No hay suficientes imagenes para {class_name}: "
                f"se necesitan {required} y hay {available}"
            )

    for split_name, target_count in split_sizes.items():
        manifest["splits"][split_name] = {}
        for class_name in CLASS_NAMES:
            class_dir = Path(output_dir) / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            downloaded = []

            while len(downloaded) < target_count and candidates[class_name]:
                image = candidates[class_name].pop()
                filename = f"{image['id']:012d}_{image['file_name']}"
                destination = class_dir / filename
                if destination.is_file() or download_image(image, destination):
                    downloaded.append(
                        {
                            "coco_id": image["id"],
                            "file": str(destination.as_posix()),
                        }
                    )

            if len(downloaded) < target_count:
                raise ValueError(
                    f"No se pudieron descargar suficientes imagenes para "
                    f"{split_name}/{class_name}"
                )
            manifest["splits"][split_name][class_name] = downloaded

    manifest_path = Path(output_dir) / "coco_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(
        description="Crea un dataset de cuatro clases usando anotaciones COCO"
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/classification"),
    )
    parser.add_argument("--images-per-class", type=int, default=400)
    parser.add_argument("--min-dominance", type=float, default=0.80)
    parser.add_argument("--min-image-area", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        manifest = create_dataset(
            annotations_path=args.annotations,
            output_dir=args.output_dir,
            images_per_class=args.images_per_class,
            min_dominance=args.min_dominance,
            min_image_area=args.min_image_area,
            seed=args.seed,
        )
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    print(f"Dataset creado en: {args.output_dir}")
    print(f"Imagenes seleccionadas: {manifest['images_per_class']} por clase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
