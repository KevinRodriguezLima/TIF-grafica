from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.piece_classification.neural_model import build_piece_transform, load_piece_classifier, torch


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in scores.values())
    if total == 0:
        return {"corner": 1 / 3, "border": 1 / 3, "interior": 1 / 3}
    return {key: max(0.0, value) / total for key, value in scores.items()}


def _aspect_ratio(piece: dict[str, Any]) -> float:
    xs = [point[0] for point in piece["vertices"]]
    ys = [point[1] for point in piece["vertices"]]
    short_side = max(1, min(max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
    return max(max(xs) - min(xs) + 1, max(ys) - min(ys) + 1) / short_side


def _length_variation(edges: list[dict[str, Any]]) -> float:
    lengths = [float(edge["length"]) for edge in edges]
    if not lengths:
        return 0.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    return variance ** 0.5 / mean


def _classify_piece_heuristic(piece: dict[str, Any], ambiguous_threshold: float) -> dict[str, Any]:
    edges = piece.get("edges", [])
    edge_count = len(edges)
    aspect = _aspect_ratio(piece)
    variation = _length_variation(edges)
    corner_score = 0.25
    border_score = 0.35
    interior_score = 0.35
    if edge_count <= 3:
        corner_score += 0.35
        border_score += 0.10
    elif edge_count == 4:
        border_score += 0.20
        interior_score += 0.10
    else:
        interior_score += 0.25
    if aspect >= 1.35:
        border_score += 0.20
        corner_score += 0.05
    else:
        interior_score += 0.10
    if variation >= 0.35:
        corner_score += 0.10
        border_score += 0.10
    else:
        interior_score += 0.05
    probabilities = _normalize({"corner": corner_score, "border": border_score, "interior": interior_score})
    predicted_class = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_class]
    return {
        "piece_id": piece["piece_id"],
        "predicted_class": "ambiguous" if confidence < ambiguous_threshold else predicted_class,
        "probabilities": probabilities,
        "confidence": confidence,
        "is_ambiguous": confidence < ambiguous_threshold,
        "features_used": {"edge_count": edge_count, "aspect_ratio": aspect, "edge_length_variation": variation},
    }


def _piece_tensor(piece: dict[str, Any], transform: Any) -> Any:
    image = cv2.imread(piece["cropped_path"], cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(piece["mask_path"], cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError(f"No se pudo leer pieza o mascara: {piece['piece_id']}")
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))
    rgba = np.dstack([rgb, mask]).astype(np.uint8)
    return transform(rgba)


def _classify_piece_model(
    piece: dict[str, Any],
    model_bundle: tuple[Any, list[str], int],
    device: str,
    ambiguous_threshold: float,
) -> dict[str, Any]:
    model, class_names, image_size = model_bundle
    transform = build_piece_transform(image_size)
    with torch.no_grad():
        tensor = _piece_tensor(piece, transform).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    probabilities = {class_names[index]: float(probs[index]) for index in range(len(class_names))}
    predicted_class = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_class]
    return {
        "piece_id": piece["piece_id"],
        "predicted_class": "ambiguous" if confidence < ambiguous_threshold else predicted_class,
        "probabilities": probabilities,
        "confidence": confidence,
        "is_ambiguous": confidence < ambiguous_threshold,
    }


def _try_load_model(model_path: Path | None, device: str) -> tuple[Any, list[str], int] | None:
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return load_piece_classifier(path, device=device)
    except (EOFError, RuntimeError, KeyError, ValueError, OSError, ImportError):
        return None


def classify_pieces(
    features_path: Path,
    output_path: Path,
    ambiguous_threshold: float = 0.50,
    model_path: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    features = json.loads(Path(features_path).read_text(encoding="utf-8"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = _try_load_model(model_path, device)

    records = []
    for piece in features.get("pieces", []):
        if model_bundle is None:
            records.append(_classify_piece_heuristic(piece, ambiguous_threshold))
        else:
            records.append(_classify_piece_model(piece, model_bundle, device, ambiguous_threshold))

    result = {
        "method": "piece_classifier_resnet18_transfer" if model_bundle else "geometry_heuristic_v1",
        "ambiguous_threshold": ambiguous_threshold,
        "classes": ["corner", "border", "interior"],
        "pieces": records,
        "errors": features.get("errors", []),
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
