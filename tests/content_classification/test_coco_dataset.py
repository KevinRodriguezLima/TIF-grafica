import json

import cv2
import numpy as np

from src.content_classification.coco_dataset import (
    calculate_split_sizes,
    create_dataset,
    group_category,
    load_candidates,
)
from src.content_classification.dataset import CLASS_NAMES, inspect_dataset


def create_coco_annotations(tmp_path):
    source_image = tmp_path / "source.png"
    cv2.imwrite(
        str(source_image),
        np.full((32, 32, 3), 120, dtype=np.uint8),
    )

    category_names = ["person", "dog", "banana", "chair"]
    categories = [
        {"id": index, "name": name}
        for index, name in enumerate(category_names, start=1)
    ]
    images = []
    annotations = []
    annotation_id = 1

    for category_id in range(1, 5):
        for item in range(3):
            image_id = category_id * 100 + item
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{image_id}.png",
                    "width": 100,
                    "height": 100,
                    "url": source_image.as_uri(),
                }
            )
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [0, 0, 80, 80],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    annotations_path = tmp_path / "instances.json"
    annotations_path.write_text(
        json.dumps(
            {
                "categories": categories,
                "images": images,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )
    return annotations_path


def test_group_category_maps_coco_names():
    assert group_category("person") == "personas"
    assert group_category("dog") == "animales"
    assert group_category("banana") == "frutas"
    assert group_category("chair") == "objetos"


def test_calculate_split_sizes_uses_seventy_fifteen_fifteen():
    assert calculate_split_sizes(400) == {
        "train": 280,
        "validation": 60,
        "test": 60,
    }


def test_load_candidates_creates_four_balanced_groups(tmp_path):
    annotations_path = create_coco_annotations(tmp_path)

    candidates = load_candidates(annotations_path, 0.60, 0.08, 42)

    assert set(candidates) == set(CLASS_NAMES)
    assert all(len(items) == 3 for items in candidates.values())


def test_create_dataset_downloads_and_splits_images(tmp_path):
    annotations_path = create_coco_annotations(tmp_path)
    output_dir = tmp_path / "dataset"

    manifest = create_dataset(
        annotations_path,
        output_dir,
        images_per_class=3,
        min_dominance=0.60,
        min_image_area=0.08,
        seed=42,
    )

    summary = inspect_dataset(output_dir)
    assert manifest["images_per_class"] == 3
    assert summary["total_images"] == 12
    for class_name in CLASS_NAMES:
        assert len(list((output_dir / "train" / class_name).glob("*.png"))) == 1
        assert len(list((output_dir / "validation" / class_name).glob("*.png"))) == 1
        assert len(list((output_dir / "test" / class_name).glob("*.png"))) == 1
