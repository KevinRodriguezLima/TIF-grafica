from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _load_piece_paths(features_path: Path) -> dict[str, str]:
    features = json.loads(Path(features_path).read_text(encoding="utf-8"))
    return {piece["piece_id"]: piece["cropped_path"] for piece in features.get("pieces", [])}


def _as_rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image.copy()


def _rotation_matrix(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    return np.array(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
        dtype=np.float64,
    )


def _transform_corners(image: np.ndarray, degrees: float, translation: list[float]) -> np.ndarray:
    height, width = image.shape[:2]
    corners = np.array(
        [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
        dtype=np.float64,
    )
    return corners @ _rotation_matrix(degrees).T + np.array(translation, dtype=np.float64)


def _warp_piece(image: np.ndarray, degrees: float, translation: list[float], offset: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    rotation = _rotation_matrix(degrees)
    target_translation = np.array(translation, dtype=np.float64) + offset
    matrix = np.array(
        [
            [rotation[0, 0], rotation[0, 1], target_translation[0]],
            [rotation[1, 0], rotation[1, 1], target_translation[1]],
        ],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        _as_rgba(image),
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _overlay_rgba(canvas: np.ndarray, layer: np.ndarray) -> int:
    alpha = layer[:, :, 3].astype(np.float32) / 255.0
    mask = alpha > 0
    overlap = int(np.count_nonzero(mask & (canvas[:, :, 3] > 0)))
    for channel in range(3):
        canvas[:, :, channel] = np.where(
            mask,
            layer[:, :, channel] * alpha + canvas[:, :, channel] * (1.0 - alpha),
            canvas[:, :, channel],
        )
    canvas[:, :, 3] = np.where(mask, 255, canvas[:, :, 3])
    return overlap


def render_reconstruction(
    assembly_path: Path,
    features_path: Path,
    reconstructed_path: Path,
    debug_path: Path,
    report_path: Path,
    padding: int = 16,
) -> dict[str, Any]:
    assembly = json.loads(Path(assembly_path).read_text(encoding="utf-8"))
    piece_paths = _load_piece_paths(features_path)

    images: dict[str, np.ndarray] = {}
    transformed_bounds: list[np.ndarray] = []
    for placement in assembly.get("placements", []):
        piece_id = placement["piece_id"]
        image = cv2.imread(piece_paths[piece_id], cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        image = _as_rgba(image)
        images[piece_id] = image
        transformed_bounds.append(
            _transform_corners(image, float(placement["rotation_degrees"]), placement["translation"])
        )

    if not transformed_bounds:
        raise ValueError("No hay piezas renderizables en la solucion de ensamblaje")

    all_points = np.vstack(transformed_bounds)
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    offset = -min_xy + padding
    width = max(1, int(math.ceil(max_xy[0] - min_xy[0] + padding * 2)))
    height = max(1, int(math.ceil(max_xy[1] - min_xy[1] + padding * 2)))
    size = (width, height)

    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    debug = np.zeros((height, width, 3), dtype=np.uint8)
    overlap_pixels = 0
    rendered = 0

    for placement in assembly.get("placements", []):
        piece_id = placement["piece_id"]
        image = images.get(piece_id)
        if image is None:
            continue
        layer = _warp_piece(image, float(placement["rotation_degrees"]), placement["translation"], offset, size)
        overlap_pixels += _overlay_rgba(canvas, layer)

        corners = _transform_corners(image, float(placement["rotation_degrees"]), placement["translation"]) + offset
        polygon = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(debug, [polygon], True, (0, 255, 0), 1)
        label_position = tuple(np.round(corners[0]).astype(int))
        cv2.putText(debug, piece_id, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        rendered += 1

    visible_pixels = int(np.count_nonzero(canvas[:, :, 3] > 0))
    total_pixels = int(canvas.shape[0] * canvas.shape[1])
    empty_area = 100.0 * (1.0 - visible_pixels / max(1, total_pixels))
    overlap = 100.0 * overlap_pixels / max(1, visible_pixels)

    reconstructed_path = Path(reconstructed_path)
    debug_path = Path(debug_path)
    report_path = Path(report_path)
    reconstructed_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(reconstructed_path), canvas)
    cv2.imwrite(str(debug_path), debug)

    report = {
        "width": int(canvas.shape[1]),
        "height": int(canvas.shape[0]),
        "pieces_rendered": rendered,
        "overlap_percentage": overlap,
        "empty_area_percentage": empty_area,
        "assembly_score": assembly.get("solution_score", 0.0),
        "reconstructed_path": str(reconstructed_path.as_posix()),
        "debug_path": str(debug_path.as_posix()),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
