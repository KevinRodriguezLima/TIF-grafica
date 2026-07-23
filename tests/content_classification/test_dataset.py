import pytest

from src.content_classification.dataset import CLASS_NAMES, SPLIT_NAMES, inspect_dataset


def _create_dataset(root, images_per_class=2):
    for split_name in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            class_dir = root / split_name / class_name
            class_dir.mkdir(parents=True)
            for index in range(images_per_class):
                (class_dir / f"image_{index}.png").write_bytes(b"fake-png")


def test_inspect_dataset_counts_images(tmp_path):
    _create_dataset(tmp_path)

    summary = inspect_dataset(tmp_path)

    assert summary["class_names"] == list(CLASS_NAMES)
    assert summary["total_images"] == len(SPLIT_NAMES) * len(CLASS_NAMES) * 2
    assert summary["splits"]["train"]["animales"] == 2


def test_inspect_dataset_rejects_missing_class(tmp_path):
    _create_dataset(tmp_path)
    missing_dir = tmp_path / "validation" / "frutas"
    for image_path in missing_dir.iterdir():
        image_path.unlink()
    missing_dir.rmdir()

    with pytest.raises(ValueError, match="validation: faltan clases: frutas"):
        inspect_dataset(tmp_path)


def test_inspect_dataset_rejects_empty_class(tmp_path):
    _create_dataset(tmp_path, images_per_class=1)
    (tmp_path / "test" / "personas" / "image_0.png").unlink()

    with pytest.raises(ValueError, match="test/personas: no contiene imagenes"):
        inspect_dataset(tmp_path)
