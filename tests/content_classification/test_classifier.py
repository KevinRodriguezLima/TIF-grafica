import json

import cv2
import numpy as np

import src.content_classification.classifier as classifier


class FakeModel:
    def predict(self, batch, verbose=0):
        assert batch.min() >= 0
        assert batch.max() <= 255
        return np.array([[0.1, 0.2, 0.6, 0.1]])


def test_keras_predictions(tmp_path, monkeypatch):
    import tensorflow as tf

    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"model")
    (tmp_path / "labels.json").write_text(
        json.dumps(["animales", "frutas", "objetos", "personas"]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        tf.keras.models,
        "load_model",
        lambda path: FakeModel(),
    )

    image = np.zeros((30, 30, 3), dtype=np.uint8)
    predictions, model_name = classifier._keras_predictions(
        image,
        model_path,
    )

    assert model_name == "mobilenetv2_transfer_learning"
    assert predictions[0]["class"] == "objetos"
    assert predictions[0]["confidence"] > 0.59


def test_classifies_with_keras_model(tmp_path, monkeypatch):
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), np.zeros((30, 30, 3), dtype=np.uint8))
    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"model")

    monkeypatch.setattr(
        classifier,
        "_model_predictions",
        lambda image, path, device: (
            [{"class": "animales", "confidence": 0.8}],
            "mobilenetv2_transfer_learning",
        ),
    )

    output_path = tmp_path / "result.json"
    result = classifier.classify_reconstructed_image(
        image_path,
        output_path,
        model_path=model_path,
    )

    assert result["accepted_prediction"] == "animales"
    assert result["model"] == "mobilenetv2_transfer_learning"


def test_classifier_cli_requires_image_and_model(monkeypatch):
    monkeypatch.setattr("sys.argv", ["classifier"])

    try:
        classifier.main()
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("Debía exigir imagen y modelo")
