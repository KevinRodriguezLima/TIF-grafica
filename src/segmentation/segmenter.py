from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _read_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _channels(image: np.ndarray) -> int:
    if image.ndim == 2:
        return 1
    return int(image.shape[2])


def _has_alpha(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[2] == 4


def build_pieces_manifest(
    pieces_dir: Path,
    output_path: Path,
    puzzle_id: str = "puzzle_001",
) -> dict[str, Any]:
    pieces_dir = Path(pieces_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not pieces_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de piezas: {pieces_dir}")

    valid_pieces: list[dict[str, Any]] = []
    invalid_pieces: list[dict[str, Any]] = []
    candidates = sorted(
        path for path in pieces_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for path in candidates:
        image = _read_image(path)
        if image is None or image.size == 0:
            invalid_pieces.append({"filename": path.name, "error": "unreadable_image"})
            logging.warning("Pieza no legible: %s", path)
            continue

        height, width = image.shape[:2]
        piece_id = f"P{len(valid_pieces):03d}"
        valid_pieces.append(
            {
                "piece_id": piece_id,
                "filename": path.name,
                "path": str(path.as_posix()),
                "width": int(width),
                "height": int(height),
                "channels": _channels(image),
                "has_alpha": _has_alpha(image),
            }
        )

    manifest = {
        "puzzle_id": puzzle_id,
        "number_of_pieces": len(valid_pieces),
        "pieces": valid_pieces,
        "invalid_pieces": invalid_pieces,
    }
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _fallback_mask_from_background(image: np.ndarray, threshold: int = 8) -> np.ndarray:
    if image.ndim == 2:
        diff = cv2.absdiff(image, np.full_like(image, int(image[0, 0])))
        return np.where(diff > threshold, 255, 0).astype(np.uint8)

    bgr = image[:, :, :3]
    corners = np.array([
        bgr[0, 0],
        bgr[0, -1],
        bgr[-1, 0],
        bgr[-1, -1],
    ], dtype=np.float32)
    background = np.median(corners, axis=0)
    distance = np.linalg.norm(bgr.astype(np.float32) - background, axis=2)
    return np.where(distance > threshold, 255, 0).astype(np.uint8)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def segment_pieces(
    manifest_path: Path,
    cropped_dir: Path,
    masks_dir: Path,
    output_path: Path,
    alpha_threshold: int = 10,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cropped_dir = Path(cropped_dir)
    masks_dir = Path(masks_dir)
    output_path = Path(output_path)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for piece in manifest.get("pieces", []):
        piece_id = piece["piece_id"]
        image_path = Path(piece["path"])
        image = _read_image(image_path)
        if image is None or image.size == 0:
            errors.append({"piece_id": piece_id, "error": "unreadable_image"})
            continue

        if _has_alpha(image):
            mask = np.where(image[:, :, 3] > alpha_threshold, 255, 0).astype(np.uint8)
        else:
            mask = _fallback_mask_from_background(image)

        mask = _clean_mask(mask)
        points = cv2.findNonZero(mask)
        if points is None:
            errors.append({"piece_id": piece_id, "error": "empty_visible_region"})
            continue

        x, y, width, height = cv2.boundingRect(points)
        cropped_image = image[y : y + height, x : x + width].copy()
        cropped_mask = mask[y : y + height, x : x + width].copy()

        if cropped_image.ndim == 2:
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGRA)
        elif cropped_image.shape[2] == 3:
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)
            cropped_image[:, :, 3] = cropped_mask
        else:
            cropped_image[:, :, 3] = np.minimum(cropped_image[:, :, 3], cropped_mask)

        cropped_path = cropped_dir / f"{piece_id}.png"
        mask_path = masks_dir / f"{piece_id}_mask.png"
        cv2.imwrite(str(cropped_path), cropped_image)
        cv2.imwrite(str(mask_path), cropped_mask)

        records.append(
            {
                "piece_id": piece_id,
                "source_path": piece["path"],
                "mask_path": str(mask_path.as_posix()),
                "cropped_path": str(cropped_path.as_posix()),
                "crop_offset": [int(x), int(y)],
                "visible_area": int(cv2.countNonZero(cropped_mask)),
                "width": int(width),
                "height": int(height),
                "status": "ok",
            }
        )

    result = {
        "alpha_threshold": alpha_threshold,
        "processed_count": len(records),
        "pieces": records,
        "errors": errors,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
