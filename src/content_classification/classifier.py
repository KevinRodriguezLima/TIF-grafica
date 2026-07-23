from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.content_classification.neural_model import build_content_transform, load_content_classifier, torch


def _as_rgb_on_background(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)


def _class_scores(image: np.ndarray) -> dict[str, float]:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    alpha = image[:, :, 3] > 0
    if not np.any(alpha):
        return {"imagen_vacia": 1.0, "personaje_animado": 0.0, "escena_natural": 0.0}
    bgr = image[:, :, :3][alpha].astype(np.float32)
    mean_bgr = bgr.mean(axis=0)
    saturation = cv2.cvtColor(bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)[:, 0, 1].mean() / 255.0
    brightness = float(mean_bgr.mean() / 255.0)
    color_spread = float(bgr.std(axis=0).mean() / 255.0)
    scores = {
        "personaje_animado": max(0.0, 0.45 * saturation + 0.35 * color_spread + 0.20 * brightness),
        "escena_natural": max(
            0.0,
            0.45 * (1.0 - abs(float(mean_bgr[1] - mean_bgr[2]) / 255.0))
            + 0.30 * brightness
            + 0.25 * (1.0 - saturation),
        ),
        "patron_abstracto": max(0.0, 0.50 * color_spread + 0.30 * (1.0 - brightness) + 0.20 * saturation),
    }
    total = sum(scores.values()) or 1.0
    return {key: value / total for key, value in scores.items()}


def _model_predictions(image: np.ndarray, model_path: Path, device: str) -> tuple[list[dict[str, float]], str]:
    model, class_names, image_size = load_content_classifier(model_path, device=device)
    transform = build_content_transform(image_size)
    rgb = _as_rgb_on_background(image)
    with torch.no_grad():
        tensor = transform(rgb).unsqueeze(0).to(device)
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    predictions = [
        {"class": class_names[index], "confidence": float(probs[index])}
        for index in range(len(class_names))
    ]
    predictions.sort(key=lambda item: item["confidence"], reverse=True)
    return predictions, "content_classifier_resnet18_transfer"


def _try_model_predictions(
    image: np.ndarray,
    model_path: Path | None,
    device: str,
) -> tuple[list[dict[str, float]], str] | None:
    if model_path is None:
        return None
    model_path = Path(model_path)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        return None
    try:
        return _model_predictions(image, model_path, device)
    except (EOFError, RuntimeError, KeyError, ValueError, OSError, ImportError):
        return None


def classify_reconstructed_image(
    image_path: Path,
    output_path: Path,
    model_name: str = "heuristic_color_classifier_v1",
    confidence_threshold: float = 0.40,
    model_path: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"No se pudo leer la imagen reconstruida: {image_path}")

    model_result = _try_model_predictions(image, model_path, device)
    if model_result is not None:
        predictions, used_model = model_result
    else:
        scores = _class_scores(image)
        predictions = [
            {"class": label, "confidence": confidence}
            for label, confidence in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]
        used_model = model_name

    accepted = predictions[0]["class"] if predictions and predictions[0]["confidence"] >= confidence_threshold else "unknown"
    result = {
        "model": used_model,
        "confidence_threshold": confidence_threshold,
        "predictions": predictions,
        "accepted_prediction": accepted,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
