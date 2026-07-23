import cv2
import numpy as np
import pytest

from src.content_classification.dataset import CLASS_NAMES, SPLIT_NAMES
from src.content_classification.trainer import build_parser, load_datasets, train


def create_images(root):
    for split_name in SPLIT_NAMES:
        for index, class_name in enumerate(CLASS_NAMES):
            class_dir = root / split_name / class_name
            class_dir.mkdir(parents=True)
            image = np.full((32, 32, 3), index * 50, dtype=np.uint8)
            cv2.imwrite(str(class_dir / "image.png"), image)


def test_load_datasets_returns_images_and_labels(tmp_path):
    create_images(tmp_path)

    train_data, validation_data, test_data = load_datasets(tmp_path, 4, 42)
    images, labels = next(iter(train_data))

    assert images.shape == (4, 224, 224, 3)
    assert labels.shape == (4,)
    assert validation_data is not None
    assert test_data is not None


def test_parser_uses_simple_defaults():
    args = build_parser().parse_args(["--dataset-dir", "data/classification"])

    assert args.epochs == 10
    assert args.batch_size == 16
    assert args.weights == "imagenet"


def test_train_rejects_invalid_values(tmp_path):
    with pytest.raises(ValueError, match="epochs"):
        train(tmp_path, tmp_path / "output", 0, 4, 1e-3, 1, 42, None)
