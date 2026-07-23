from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_compatibility_graph(
    filtered_pairs_path: Path,
    classification_path: Path,
    output_path: Path,
    top_k_per_edge: int = 5,
) -> dict[str, Any]:
    pairs_data = json.loads(Path(filtered_pairs_path).read_text(encoding="utf-8"))
    classification = json.loads(Path(classification_path).read_text(encoding="utf-8"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = [
        {
            "piece_id": piece["piece_id"],
            "class": piece["predicted_class"],
            "confidence": piece["confidence"],
        }
        for piece in classification.get("pieces", [])
    ]

    by_edge: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs_data.get("pairs", []):
        by_edge.setdefault(pair["edge_a"], []).append(pair)
        by_edge.setdefault(pair["edge_b"], []).append(pair)

    allowed_ids: set[str] = set()
    for edge_pairs in by_edge.values():
        edge_pairs.sort(key=lambda item: item["score"], reverse=True)
        allowed_ids.update(pair["candidate_id"] for pair in edge_pairs[:top_k_per_edge])

    graph_edges = []
    for pair in pairs_data.get("pairs", []):
        if pair["candidate_id"] not in allowed_ids:
            continue
        graph_edges.append(
            {
                "candidate_id": pair["candidate_id"],
                "piece_a": pair["piece_a"],
                "edge_a": pair["edge_a"],
                "piece_b": pair["piece_b"],
                "edge_b": pair["edge_b"],
                "score": pair["score"],
                "relative_rotation_degrees": pair["relative_rotation_degrees"],
                "translation": pair["translation"],
                "edge_a_geometry": pair["edge_a_geometry"],
                "edge_b_geometry": pair["edge_b_geometry"],
            }
        )

    graph_edges.sort(key=lambda item: item["score"], reverse=True)
    result = {
        "method": "top_k_edge_graph_v1",
        "top_k_per_edge": top_k_per_edge,
        "nodes": nodes,
        "edges": graph_edges,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
