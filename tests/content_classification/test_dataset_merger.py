import cv2
import numpy as np

from src.content_classification.dataset import CLASS_NAMES
from src.content_classification.dataset import inspect_dataset
from src.content_classification.dataset_merger import merge_datasets


def create_image(path, seed):
    generator = np.random.default_rng(seed)
    image = generator.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def test_merge_datasets(tmp_path):
    coco_dir = tmp_path / "coco"
    openverse_dir = tmp_path / "openverse"

    for class_number, class_name in enumerate(CLASS_NAMES):
        for number in range(3):
            create_image(
                coco_dir / "train" / class_name / f"coco_{number}.jpg",
                class_number * 100 + number,
            )
        for number in range(3):
            create_image(
                openverse_dir / class_name / f"openverse_{number}.jpg",
                class_number * 100 + number + 20,
            )

    duplicate = openverse_dir / "animales" / "duplicate.jpg"
    duplicate.write_bytes(
        (coco_dir / "train" / "animales" / "coco_0.jpg").read_bytes()
    )

    output_dir = tmp_path / "merged"
    manifest = merge_datasets(
        [coco_dir, openverse_dir],
        output_dir,
        images_per_class=4,
        seed=42,
    )
    summary = inspect_dataset(output_dir)

    assert manifest["duplicates_removed"] == 1
    assert summary["total_images"] == 16
    assert summary["splits"]["train"]["animales"] == 2
    assert summary["splits"]["validation"]["animales"] == 1
    assert summary["splits"]["test"]["animales"] == 1


def test_merge_requires_enough_images(tmp_path):
    source_dir = tmp_path / "source"
    for class_number, class_name in enumerate(CLASS_NAMES):
        create_image(
            source_dir / class_name / "image.jpg",
            class_number,
        )

    try:
        merge_datasets(
            [source_dir],
            tmp_path / "output",
            images_per_class=3,
            seed=42,
        )
    except ValueError as error:
        assert "No hay suficientes imágenes" in str(error)
    else:
        raise AssertionError("Debía rechazar un dataset insuficiente")
