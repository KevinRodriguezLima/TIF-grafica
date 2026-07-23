from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _load_piece_images(features_path: Path) -> dict[str, np.ndarray]:
    features = json.loads(Path(features_path).read_text(encoding="utf-8"))
    images: dict[str, np.ndarray] = {}
    for piece in features.get("pieces", []):
        image = cv2.imread(piece["cropped_path"], cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        images[piece["piece_id"]] = image
    return images


def _rotation_matrix(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)
    return np.array([[cos_v, -sin_v], [sin_v, cos_v]], dtype=np.float64)


def _transform_point(point: list[float], rotation_degrees: float, translation: np.ndarray) -> np.ndarray:
    return _rotation_matrix(rotation_degrees) @ np.array(point, dtype=np.float64) + translation


def _angle_difference_degrees(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _piece_corners(image: np.ndarray) -> list[list[float]]:
    height, width = image.shape[:2]
    return [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]


def _placement_bounds(piece_id: str, placement: dict[str, Any], images: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    transformed = [
        _transform_point(
            corner,
            float(placement["rotation_degrees"]),
            np.array(placement["translation"], dtype=np.float64),
        )
        for corner in _piece_corners(images[piece_id])
    ]
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    return min(xs), min(ys), max(xs), max(ys)


def _warp_alpha_mask(
    image: np.ndarray,
    rotation_degrees: float,
    translation: list[float],
    offset: np.ndarray,
    size: tuple[int, int],
    scale: float,
) -> np.ndarray:
    alpha = image[:, :, 3]
    rotation = _rotation_matrix(rotation_degrees)
    target_translation = (np.array(translation, dtype=np.float64) + offset) * scale
    matrix = np.array(
        [
            [rotation[0, 0] * scale, rotation[0, 1] * scale, target_translation[0]],
            [rotation[1, 0] * scale, rotation[1, 1] * scale, target_translation[1]],
        ],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        alpha,
        matrix,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0


def _actual_overlap_ratio(
    candidate_piece_id: str,
    candidate: dict[str, Any],
    placements: dict[str, dict[str, Any]],
    images: dict[str, np.ndarray],
    scale: float = 0.35,
) -> float:
    if not placements:
        return 0.0

    all_bounds = [_placement_bounds(candidate_piece_id, candidate, images)]
    all_bounds.extend(_placement_bounds(piece_id, placement, images) for piece_id, placement in placements.items())
    min_x = min(bounds[0] for bounds in all_bounds)
    min_y = min(bounds[1] for bounds in all_bounds)
    max_x = max(bounds[2] for bounds in all_bounds)
    max_y = max(bounds[3] for bounds in all_bounds)
    offset = np.array([-min_x + 2.0, -min_y + 2.0], dtype=np.float64)
    width = max(1, int(math.ceil((max_x - min_x + 4.0) * scale)))
    height = max(1, int(math.ceil((max_y - min_y + 4.0) * scale)))
    size = (width, height)

    occupied = np.zeros((height, width), dtype=bool)
    for piece_id, placement in placements.items():
        occupied |= _warp_alpha_mask(
            images[piece_id],
            float(placement["rotation_degrees"]),
            placement["translation"],
            offset,
            size,
            scale,
        )

    candidate_mask = _warp_alpha_mask(
        images[candidate_piece_id],
        float(candidate["rotation_degrees"]),
        candidate["translation"],
        offset,
        size,
        scale,
    )
    candidate_area = int(np.count_nonzero(candidate_mask))
    if candidate_area == 0:
        return 1.0
    overlap = int(np.count_nonzero(candidate_mask & occupied))
    return overlap / candidate_area


def _place_unknown_from_known(
    match: dict[str, Any],
    known_piece_key: str,
    known_edge_key: str,
    unknown_piece_key: str,
    unknown_edge_key: str,
    placements: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    known_piece = match[known_piece_key]
    unknown_piece = match[unknown_piece_key]
    known_edge = match[known_edge_key]
    unknown_edge = match[unknown_edge_key]

    known_transform = placements[known_piece]
    known_rotation = float(known_transform["rotation_degrees"])
    known_translation = np.array(known_transform["translation"], dtype=np.float64)

    target_rotation = (
        known_rotation
        + float(known_edge["angle_degrees"])
        + 180.0
        - float(unknown_edge["angle_degrees"])
    ) % 360.0

    known_midpoint = _transform_point(known_edge["midpoint"], known_rotation, known_translation)
    unknown_midpoint_rotated = _rotation_matrix(target_rotation) @ np.array(unknown_edge["midpoint"], dtype=np.float64)
    target_translation = known_midpoint - unknown_midpoint_rotated

    return {
        "piece_id": unknown_piece,
        "rotation_degrees": target_rotation,
        "translation": [float(target_translation[0]), float(target_translation[1])],
    }


def _component_bounds(
    placements: dict[str, dict[str, Any]],
    component_pieces: list[str],
    images: dict[str, np.ndarray],
) -> tuple[float, float, float, float]:
    bounds = [_placement_bounds(piece_id, placements[piece_id], images) for piece_id in component_pieces]
    return (
        min(bound[0] for bound in bounds),
        min(bound[1] for bound in bounds),
        max(bound[2] for bound in bounds),
        max(bound[3] for bound in bounds),
    )


def _layout_components(
    placements: dict[str, dict[str, Any]],
    components: dict[str, int],
    images: dict[str, np.ndarray],
    grid_gap: int,
) -> list[dict[str, Any]]:
    by_component: dict[int, list[str]] = {}
    for piece_id, component_id in components.items():
        by_component.setdefault(component_id, []).append(piece_id)

    arranged: list[dict[str, Any]] = []
    cursor_x = 0.0
    for component_id in sorted(by_component):
        component_pieces = sorted(piece_id for piece_id in by_component[component_id] if piece_id in images)
        if not component_pieces:
            continue
        min_x, min_y, max_x, _ = _component_bounds(placements, component_pieces, images)
        component_offset = np.array([cursor_x - min_x, -min_y], dtype=np.float64)
        for piece_id in component_pieces:
            placement = placements[piece_id]
            translation = np.array(placement["translation"], dtype=np.float64) + component_offset
            arranged.append(
                {
                    "piece_id": piece_id,
                    "rotation_degrees": float(placement["rotation_degrees"]),
                    "translation": [float(translation[0]), float(translation[1])],
                    "component_id": component_id,
                }
            )
        cursor_x += max_x - min_x + grid_gap
    return sorted(arranged, key=lambda item: item["piece_id"])


def _place_unmatched_pieces(
    piece_ids: list[str],
    placements: dict[str, dict[str, Any]],
    components: dict[str, int],
    images: dict[str, np.ndarray],
    grid_gap: int,
    next_component_id: int,
) -> int:
    missing = [piece_id for piece_id in piece_ids if piece_id not in placements]
    cursor_x = 0.0
    for piece_id in missing:
        placements[piece_id] = {
            "piece_id": piece_id,
            "rotation_degrees": 0.0,
            "translation": [cursor_x, 0.0],
        }
        components[piece_id] = next_component_id
        next_component_id += 1
        cursor_x += images[piece_id].shape[1] + grid_gap
    return next_component_id


def solve_assembly(
    graph_path: Path,
    features_path: Path,
    output_path: Path,
    grid_gap: int = 24,
    consistency_tolerance: float = 50.0,
    max_overlap_ratio: float = 0.20,
) -> dict[str, Any]:
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = _load_piece_images(features_path)
    piece_ids = sorted(images)
    placements: dict[str, dict[str, Any]] = {}
    components: dict[str, int] = {}
    selected_matches: list[dict[str, Any]] = []
    rejected_matches: list[dict[str, Any]] = []
    used_edges: set[str] = set()
    next_component_id = 0

    for edge in graph.get("edges", []):
        piece_a = edge["piece_a"]
        piece_b = edge["piece_b"]
        if piece_a not in images or piece_b not in images:
            continue
        if edge["edge_a"] in used_edges or edge["edge_b"] in used_edges:
            continue

        candidate: dict[str, Any] | None = None
        candidate_piece: str | None = None
        started_component = False

        if piece_a not in placements and piece_b not in placements:
            component_id = next_component_id
            next_component_id += 1
            placements[piece_a] = {"piece_id": piece_a, "rotation_degrees": 0.0, "translation": [0.0, 0.0]}
            components[piece_a] = component_id
            components[piece_b] = component_id
            candidate = _place_unknown_from_known(
                edge, "piece_a", "edge_a_geometry", "piece_b", "edge_b_geometry", placements
            )
            candidate_piece = piece_b
            started_component = True
        elif piece_a in placements and piece_b not in placements:
            candidate = _place_unknown_from_known(
                edge, "piece_a", "edge_a_geometry", "piece_b", "edge_b_geometry", placements
            )
            candidate_piece = piece_b
            components[piece_b] = components[piece_a]
        elif piece_b in placements and piece_a not in placements:
            candidate = _place_unknown_from_known(
                edge, "piece_b", "edge_b_geometry", "piece_a", "edge_a_geometry", placements
            )
            candidate_piece = piece_a
            components[piece_a] = components[piece_b]
        else:
            current_a = placements[piece_a]
            current_b = placements[piece_b]
            midpoint_a = _transform_point(
                edge["edge_a_geometry"]["midpoint"],
                current_a["rotation_degrees"],
                np.array(current_a["translation"], dtype=np.float64),
            )
            midpoint_b = _transform_point(
                edge["edge_b_geometry"]["midpoint"],
                current_b["rotation_degrees"],
                np.array(current_b["translation"], dtype=np.float64),
            )
            distance = float(np.linalg.norm(midpoint_a - midpoint_b))
            angle_a = current_a["rotation_degrees"] + edge["edge_a_geometry"]["angle_degrees"]
            angle_b = current_b["rotation_degrees"] + edge["edge_b_geometry"]["angle_degrees"]
            angle_error = _angle_difference_degrees((angle_a + 180.0) % 360.0, angle_b)
            if distance > consistency_tolerance or angle_error > 20.0:
                rejected_matches.append({"candidate_id": edge["candidate_id"], "reason": "inconsistent_existing"})
                continue

        if candidate is not None and candidate_piece is not None:
            overlap_ratio = _actual_overlap_ratio(candidate_piece, candidate, placements, images)
            if overlap_ratio > max_overlap_ratio:
                rejected_matches.append(
                    {
                        "candidate_id": edge["candidate_id"],
                        "reason": "overlap",
                        "overlap_ratio": overlap_ratio,
                    }
                )
                if started_component:
                    placements.pop(piece_a, None)
                    components.pop(piece_a, None)
                    components.pop(piece_b, None)
                else:
                    components.pop(candidate_piece, None)
                continue
            placements[candidate_piece] = candidate

        selected_matches.append(
            {
                "candidate_id": edge["candidate_id"],
                "edge_a": edge["edge_a"],
                "edge_b": edge["edge_b"],
                "score": edge["score"],
                "piece_a": piece_a,
                "piece_b": piece_b,
            }
        )
        used_edges.add(edge["edge_a"])
        used_edges.add(edge["edge_b"])

    next_component_id = _place_unmatched_pieces(piece_ids, placements, components, images, grid_gap, next_component_id)
    normalized = _layout_components(placements, components, images, grid_gap)
    placed_ids = {item["piece_id"] for item in normalized}
    average_score = sum(match["score"] for match in selected_matches) / max(1, len(selected_matches))
    result = {
        "method": "overlap_aware_edge_transform_solver_v1",
        "solution_score": average_score,
        "max_overlap_ratio": max_overlap_ratio,
        "pieces_used": len(normalized),
        "unplaced_pieces": [piece_id for piece_id in piece_ids if piece_id not in placed_ids],
        "placements": normalized,
        "selected_matches": selected_matches,
        "rejected_matches": rejected_matches,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result

