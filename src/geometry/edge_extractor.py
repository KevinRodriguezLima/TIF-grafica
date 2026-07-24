from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _signed_area(points: list[list[int]]) -> float:
    area = 0.0
    total = len(points)
    for index in range(total):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % total]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _ensure_clockwise(points: list[list[int]]) -> list[list[int]]:
    if len(points) >= 3 and _signed_area(points) < 0:
        return list(reversed(points))
    return points


def _remove_short_edges(vertices: list[list[int]], min_length: float) -> list[list[int]]:
    if len(vertices) <= 3:
        return vertices

    filtered: list[list[int]] = []
    for vertex in vertices:
        if not filtered:
            filtered.append(vertex)
            continue
        previous = np.array(filtered[-1], dtype=np.float32)
        current = np.array(vertex, dtype=np.float32)
        if float(np.linalg.norm(current - previous)) >= min_length:
            filtered.append(vertex)

    if len(filtered) > 3:
        first = np.array(filtered[0], dtype=np.float32)
        last = np.array(filtered[-1], dtype=np.float32)
        if float(np.linalg.norm(first - last)) < min_length:
            filtered.pop()
    return filtered if len(filtered) >= 3 else vertices


def _write_edge_strip(
    image: np.ndarray,
    edge_id: str,
    start: np.ndarray,
    end: np.ndarray,
    inward: np.ndarray,
    strip_width: int,
    output_dir: Path,
) -> str:
    height, width = image.shape[:2]
    strip_polygon = np.array(
        [
            start,
            end,
            end + inward * strip_width,
            start + inward * strip_width,
        ],
        dtype=np.float32,
    )
    strip_polygon[:, 0] = np.clip(strip_polygon[:, 0], 0, width - 1)
    strip_polygon[:, 1] = np.clip(strip_polygon[:, 1], 0, height - 1)
    strip_polygon_i = np.round(strip_polygon).astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [strip_polygon_i], 255)

    if image.ndim == 2:
        rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        rgba = image.copy()
    rgba[:, :, 3] = np.minimum(rgba[:, :, 3], mask)

    x, y, w, h = cv2.boundingRect(strip_polygon_i)
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    strip = rgba[y : y + h, x : x + w]

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{edge_id}.png"
    cv2.imwrite(str(path), strip)
    return str(path.as_posix())


def detect_edges(
    contours_path: Path,
    output_path: Path,
    edge_strips_dir: Path,
    epsilon_factor: float = 0.01,
    min_edge_length: float = 5.0,
    strip_width: int = 16,
) -> dict[str, Any]:
    contours_data = json.loads(Path(contours_path).read_text(encoding="utf-8"))
    output_path = Path(output_path)
    edge_strips_dir = Path(edge_strips_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_strips_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for piece in contours_data.get("pieces", []):
        piece_id = piece["piece_id"]
        contour = np.array(piece["contour_points"], dtype=np.int32).reshape(-1, 1, 2)
        perimeter = float(piece["perimeter"])
        epsilon = float(epsilon_factor * perimeter)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        vertices = polygon.reshape(-1, 2).astype(int).tolist()
        vertices = _ensure_clockwise(vertices)
        vertices = _remove_short_edges(vertices, min_edge_length)

        if len(vertices) < 3:
            errors.append({"piece_id": piece_id, "error": "not_enough_vertices"})
            continue

        image = cv2.imread(str(Path(piece["cropped_path"])), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            errors.append({"piece_id": piece_id, "error": "unreadable_cropped_image"})
            continue

        edges: list[dict[str, Any]] = []
        for index, start_values in enumerate(vertices):
            end_values = vertices[(index + 1) % len(vertices)]
            start = np.array(start_values, dtype=np.float32)
            end = np.array(end_values, dtype=np.float32)
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length == 0:
                continue

            direction = vector / length
            inward = np.array([-direction[1], direction[0]], dtype=np.float32)
            midpoint = (start + end) / 2.0
            angle = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
            edge_id = f"{piece_id}_E{index}"
            strip_path = _write_edge_strip(
                image,
                edge_id,
                start,
                end,
                inward,
                strip_width,
                edge_strips_dir,
            )

            edges.append(
                {
                    "edge_id": edge_id,
                    "start": [int(start[0]), int(start[1])],
                    "end": [int(end[0]), int(end[1])],
                    "length": length,
                    "angle_degrees": angle,
                    "midpoint": [float(midpoint[0]), float(midpoint[1])],
                    "direction": [float(direction[0]), float(direction[1])],
                    "inward_normal": [float(inward[0]), float(inward[1])],
                    "strip_path": strip_path,
                }
            )

        records.append(
            {
                "piece_id": piece_id,
                "vertices": vertices,
                "edges": edges,
                "edge_count": len(edges),
                "epsilon": epsilon,
                "perimeter": perimeter,
                "cropped_path": piece["cropped_path"],
                "mask_path": piece["mask_path"],
            }
        )

    result = {
        "epsilon_factor": epsilon_factor,
        "min_edge_length": min_edge_length,
        "strip_width": strip_width,
        "pieces": records,
        "errors": errors,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
