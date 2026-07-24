import cv2
import numpy as np

from src.rendering.reconstruction_evaluator import evaluate_reconstruction


def test_evaluates_identical_reconstruction(tmp_path):
    reference = np.full((30, 30, 3), 120, dtype=np.uint8)
    reconstruction = np.zeros((40, 40, 4), dtype=np.uint8)
    reconstruction[5:35, 5:35, :3] = reference
    reconstruction[5:35, 5:35, 3] = 255

    reference_path = tmp_path / "reference.png"
    reconstruction_path = tmp_path / "reconstruction.png"
    output_path = tmp_path / "evaluation.json"
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(reconstruction_path), reconstruction)

    result = evaluate_reconstruction(
        reference_path,
        reconstruction_path,
        output_path,
    )

    assert result["similarity"] == 1.0
    assert result["mean_absolute_error"] == 0.0
    assert output_path.is_file()
