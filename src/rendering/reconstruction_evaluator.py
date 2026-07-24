import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def crop_reconstruction(image):
    if image.ndim == 3 and image.shape[2] == 4:
        mask = image[:, :, 3] > 0
        if not np.any(mask):
            raise ValueError("La reconstrucción está vacía")
        y_values, x_values = np.where(mask)
        image = image[
            y_values.min() : y_values.max() + 1,
            x_values.min() : x_values.max() + 1,
            :3,
        ]
    return image


def evaluate_reconstruction(reference_path, reconstruction_path, output_path):
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    reconstruction = cv2.imread(
        str(reconstruction_path),
        cv2.IMREAD_UNCHANGED,
    )
    if reference is None:
        raise FileNotFoundError(f"No se pudo leer: {reference_path}")
    if reconstruction is None:
        raise FileNotFoundError(f"No se pudo leer: {reconstruction_path}")

    reconstruction = crop_reconstruction(reconstruction)
    original_size = [
        int(reconstruction.shape[1]),
        int(reconstruction.shape[0]),
    ]
    reconstruction = cv2.resize(
        reconstruction,
        (reference.shape[1], reference.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    difference = reference.astype(np.float32) - reconstruction.astype(
        np.float32
    )
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(difference ** 2))
    psnr = 100.0 if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
    similarity = max(0.0, 1.0 - mae / 255.0)

    result = {
        "reference_size": [
            int(reference.shape[1]),
            int(reference.shape[0]),
        ],
        "reconstruction_content_size": original_size,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "psnr": psnr,
        "similarity": similarity,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = evaluate_reconstruction(
            args.reference,
            args.reconstruction,
            args.output,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
