from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in scores.values())
    if total == 0:
        return {"corner": 1 / 3, "border": 1 / 3, "interior": 1 / 3}
    return {key: max(0.0, value) / total for key, value in scores.items()}


def _aspect_ratio(piece: dict[str, Any]) -> float:
    widths = [point[0] for point in piece["vertices"]]
    heights = [point[1] for point in piece["vertices"]]
    width = max(widths) - min(widths) + 1
    height = max(heights) - min(heights) + 1
    short_side = max(1, min(width, height))
    return max(width, height) / short_side


def _length_variation(edges: list[dict[str, Any]]) -> float:
    lengths = [float(edge["length"]) for edge in edges]
    if not lengths:
        return 0.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    return variance ** 0.5 / mean


def classify_piece(piece: dict[str, Any], ambiguous_threshold: float) -> dict[str, Any]:
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

    probabilities = _normalize(
        {
            "corner": corner_score,
            "border": border_score,
            "interior": interior_score,
        }
    )
    predicted_class = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_class]
    is_ambiguous = confidence < ambiguous_threshold

    return {
        "piece_id": piece["piece_id"],
        "predicted_class": "ambiguous" if is_ambiguous else predicted_class,
        "probabilities": probabilities,
        "confidence": confidence,
        "is_ambiguous": is_ambiguous,
        "features_used": {
            "edge_count": edge_count,
            "aspect_ratio": aspect,
            "edge_length_variation": variation,
        },
    }


def classify_pieces(
    features_path: Path,
    output_path: Path,
    ambiguous_threshold: float = 0.50,
) -> dict[str, Any]:
    features = json.loads(Path(features_path).read_text(encoding="utf-8"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        classify_piece(piece, ambiguous_threshold)
        for piece in features.get("pieces", [])
    ]
    result = {
        "method": "geometry_heuristic_v1",
        "ambiguous_threshold": ambiguous_threshold,
        "classes": ["corner", "border", "interior"],
        "pieces": records,
        "errors": features.get("errors", []),
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
