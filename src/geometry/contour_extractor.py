from __future__ import annotations

import json
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


def _ordered_clockwise(points: list[list[int]]) -> list[list[int]]:
    # In image coordinates, positive signed area follows the clockwise visual order.
    if len(points) >= 3 and _signed_area(points) < 0:
        return list(reversed(points))
    return points


def extract_contours(
    segmentation_path: Path,
    output_path: Path,
    contours_dir: Path | None = None,
    min_area: float = 10.0,
) -> dict[str, Any]:
    segmentation = json.loads(Path(segmentation_path).read_text(encoding="utf-8"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if contours_dir is not None:
        Path(contours_dir).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for piece in segmentation.get("pieces", []):
        piece_id = piece["piece_id"]
        mask_path = Path(piece["mask_path"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.size == 0:
            errors.append({"piece_id": piece_id, "error": "unreadable_mask"})
            continue

        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            errors.append({"piece_id": piece_id, "error": "missing_contour"})
            continue

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < min_area:
            errors.append({"piece_id": piece_id, "error": "contour_too_small"})
            continue

        perimeter = float(cv2.arcLength(contour, True))
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            centroid = [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]
        else:
            centroid = [0.0, 0.0]

        points = contour.reshape(-1, 2).astype(int).tolist()
        points = _ordered_clockwise(points)
        orientation = "clockwise"
        x, y, width, height = cv2.boundingRect(np.array(points, dtype=np.int32))

        debug_path = None
        if contours_dir is not None:
            debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug, [np.array(points, dtype=np.int32).reshape(-1, 1, 2)], -1, (0, 0, 255), 1)
            debug_file = Path(contours_dir) / f"{piece_id}_contour.png"
            cv2.imwrite(str(debug_file), debug)
            debug_path = str(debug_file.as_posix())

        record = {
            "piece_id": piece_id,
            "mask_path": piece["mask_path"],
            "cropped_path": piece["cropped_path"],
            "area": area,
            "perimeter": perimeter,
            "centroid": centroid,
            "is_convex": bool(cv2.isContourConvex(np.array(points, dtype=np.int32).reshape(-1, 1, 2))),
            "orientation": orientation,
            "bounding_box": [int(x), int(y), int(width), int(height)],
            "contour_points": points,
        }
        if debug_path is not None:
            record["contour_debug_path"] = debug_path
        records.append(record)

    result = {"pieces": records, "errors": errors}
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
