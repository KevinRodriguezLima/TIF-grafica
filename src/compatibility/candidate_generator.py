from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _piece_classes(classification: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        piece["piece_id"]: piece
        for piece in classification.get("pieces", [])
    }


def _collect_edges(features: dict[str, Any]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for piece in features.get("pieces", []):
        piece_id = piece["piece_id"]
        for edge in piece.get("edges", []):
            edges.append(
                {
                    "piece_id": piece_id,
                    "edge_id": edge["edge_id"],
                    "strip_path": edge["strip_path"],
                    "start": edge["start"],
                    "end": edge["end"],
                    "length": edge["length"],
                    "angle_degrees": edge["angle_degrees"],
                    "midpoint": edge["midpoint"],
                    "direction": edge["direction"],
                    "inward_normal": edge["inward_normal"],
                }
            )
    return edges


def generate_candidate_pairs(
    features_path: Path,
    classification_path: Path,
    output_path: Path,
    max_length_ratio: float | None = None,
) -> dict[str, Any]:
    features = json.loads(Path(features_path).read_text(encoding="utf-8"))
    classification = json.loads(Path(classification_path).read_text(encoding="utf-8"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = _piece_classes(classification)
    edges = _collect_edges(features)
    candidates: list[dict[str, Any]] = []
    skipped_same_piece = 0
    skipped_length_ratio = 0

    for index_a, edge_a in enumerate(edges):
        for edge_b in edges[index_a + 1:]:
            if edge_a["piece_id"] == edge_b["piece_id"]:
                skipped_same_piece += 1
                continue

            length_a = float(edge_a["length"])
            length_b = float(edge_b["length"])
            length_ratio = max(length_a, length_b) / max(1e-9, min(length_a, length_b))
            if max_length_ratio is not None and length_ratio > max_length_ratio:
                skipped_length_ratio += 1
                continue

            candidate_id = f"C{len(candidates):06d}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "piece_a": edge_a["piece_id"],
                    "edge_a": edge_a["edge_id"],
                    "piece_b": edge_b["piece_id"],
                    "edge_b": edge_b["edge_id"],
                    "strip_a": edge_a["strip_path"],
                    "strip_b": edge_b["strip_path"],
                    "piece_a_class": classes.get(edge_a["piece_id"], {}).get("predicted_class"),
                    "piece_b_class": classes.get(edge_b["piece_id"], {}).get("predicted_class"),
                    "edge_a_geometry": {
                        "start": edge_a["start"],
                        "end": edge_a["end"],
                        "length": edge_a["length"],
                        "angle_degrees": edge_a["angle_degrees"],
                        "midpoint": edge_a["midpoint"],
                        "direction": edge_a["direction"],
                        "inward_normal": edge_a["inward_normal"],
                    },
                    "edge_b_geometry": {
                        "start": edge_b["start"],
                        "end": edge_b["end"],
                        "length": edge_b["length"],
                        "angle_degrees": edge_b["angle_degrees"],
                        "midpoint": edge_b["midpoint"],
                        "direction": edge_b["direction"],
                        "inward_normal": edge_b["inward_normal"],
                    },
                    "length_ratio": length_ratio,
                }
            )

    result = {
        "total_edges": len(edges),
        "candidate_count": len(candidates),
        "max_length_ratio": max_length_ratio,
        "skipped_same_piece": skipped_same_piece,
        "skipped_length_ratio": skipped_length_ratio,
        "candidates": candidates,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
