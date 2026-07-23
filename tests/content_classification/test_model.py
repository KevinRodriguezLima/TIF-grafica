import numpy as np
import pytest

from src.content_classification.dataset import CLASS_NAMES
from src.content_classification.model import (
    IMAGE_SIZE,
    build_mobilenetv2,
    compile_model,
    model_metadata,
)


@pytest.fixture(scope="module")
def model_and_base():
    return build_mobilenetv2(weights=None)


def test_model_has_expected_input_and_output(model_and_base):
    model, base_model = model_and_base

    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, len(CLASS_NAMES))
    assert base_model.trainable is False


def test_model_outputs_probabilities(model_and_base):
    model, _ = model_and_base
    image = np.zeros((1, *IMAGE_SIZE, 3), dtype=np.float32)

    probabilities = model(image, training=False).numpy()

    assert probabilities.shape == (1, len(CLASS_NAMES))
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0], atol=1e-6)


def test_compile_model_configures_training(model_and_base):
    model, _ = model_and_base

    compile_model(model)

    assert model.optimizer is not None
    assert model.loss is not None


def test_invalid_hyperparameters_are_rejected(model_and_base):
    model, _ = model_and_base

    with pytest.raises(ValueError, match="dropout_rate"):
        build_mobilenetv2(weights=None, dropout_rate=1.0)
    with pytest.raises(ValueError, match="learning_rate"):
        compile_model(model, learning_rate=0)


def test_model_metadata_describes_contract():
    assert model_metadata() == {
        "architecture": "MobileNetV2",
        "input_size": [224, 224],
        "channels": 3,
        "class_names": list(CLASS_NAMES),
        "pretrained_weights": "imagenet",
    }
