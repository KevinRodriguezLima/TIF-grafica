from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.compatibility.siamese_model import build_edge_transform, load_siamese_model, torch


def _angle_difference_degrees(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _opposite_angle_error(a: float, b: float) -> float:
    return _angle_difference_degrees((a + 180.0) % 360.0, b)


def _as_bgra(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def _strip_profile(path: str, samples: int = 32) -> np.ndarray | None:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None
    image = _as_bgra(image)
    alpha = image[:, :, 3] > 0
    if not np.any(alpha):
        return None

    columns = []
    for x in range(image.shape[1]):
        mask = alpha[:, x]
        if np.any(mask):
            columns.append(image[:, x, :3][mask].mean(axis=0))
    if not columns:
        return None
    profile = np.array(columns, dtype=np.float32) / 255.0
    indexes = np.linspace(0, len(profile) - 1, samples).astype(int)
    return profile[indexes].reshape(-1)


def _strip_signature(path: str) -> np.ndarray | None:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None
    image = _as_bgra(image)
    alpha = image[:, :, 3] > 0
    if not np.any(alpha):
        return None
    rgb = image[:, :, :3][alpha].astype(np.float32) / 255.0
    return np.concatenate([rgb.mean(axis=0), rgb.std(axis=0)])


def _visual_score(strip_a: str, strip_b: str) -> float:
    profile_a = _strip_profile(strip_a)
    profile_b = _strip_profile(strip_b)
    signature_a = _strip_signature(strip_a)
    signature_b = _strip_signature(strip_b)
    if profile_a is None or profile_b is None or signature_a is None or signature_b is None:
        return 0.5
    profile_b_reversed = profile_b.reshape(-1, 3)[::-1].reshape(-1)
    profile_distance = float(np.mean(np.abs(profile_a - profile_b_reversed)))
    signature_distance = float(np.linalg.norm(signature_a - signature_b) / math.sqrt(len(signature_a)))
    return max(0.0, min(1.0, 1.0 - (0.70 * profile_distance + 0.30 * signature_distance)))


def _read_strip_rgb(path: str) -> np.ndarray | None:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None
    image = _as_bgra(image)
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    rgb = image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)


class SiameseEdgeScorer:
    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        if torch is None:
            raise ImportError("torch no esta instalado")
        checkpoint = torch.load(str(model_path), map_location=device)
        self.image_size = int(checkpoint.get("image_size", 224)) if isinstance(checkpoint, dict) else 224
        embedding_dim = int(checkpoint.get("embedding_dim", 128)) if isinstance(checkpoint, dict) else 128
        self.device = device
        self.model = load_siamese_model(model_path, device=device, embedding_dim=embedding_dim)
        self.transform = build_edge_transform(self.image_size)

    def score(self, strip_a: str, strip_b: str) -> float:
        image_a = _read_strip_rgb(strip_a)
        image_b = _read_strip_rgb(strip_b)
        if image_a is None or image_b is None:
            return 0.5
        with torch.no_grad():
            tensor_a = self.transform(image_a).unsqueeze(0).to(self.device)
            tensor_b = self.transform(image_b).unsqueeze(0).to(self.device)
            emb_a, emb_b = self.model(tensor_a, tensor_b)
            distance = torch.nn.functional.pairwise_distance(emb_a, emb_b).item()
        return max(0.0, min(1.0, 1.0 - distance))


def _try_load_siamese(model_path: Path | None, device: str) -> SiameseEdgeScorer | None:
    if model_path is None:
        return None
    model_path = Path(model_path)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        return None
    try:
        return SiameseEdgeScorer(model_path, device)
    except (EOFError, RuntimeError, KeyError, ValueError, OSError, ImportError):
        return None


def filter_and_score_pairs(
    candidate_pairs_path: Path,
    output_path: Path,
    max_length_ratio: float = 1.45,
    max_opposite_angle_error: float = 25.0,
    min_score: float = 0.55,
    siamese_model_path: Path | None = None,
    siamese_device: str = "cpu",
) -> dict[str, Any]:
    data = json.loads(Path(candidate_pairs_path).read_text(encoding="utf-8"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    siamese_scorer = _try_load_siamese(siamese_model_path, siamese_device)

    for candidate in data.get("candidates", []):
        length_ratio = float(candidate.get("length_ratio", 1.0))
        angle_a = float(candidate["edge_a_geometry"]["angle_degrees"])
        angle_b = float(candidate["edge_b_geometry"]["angle_degrees"])
        angle_error = _opposite_angle_error(angle_a, angle_b)
        reasons = []
        if length_ratio > max_length_ratio:
            reasons.append("length_ratio")
        if angle_error > max_opposite_angle_error:
            reasons.append("opposite_angle")
        if reasons:
            rejected.append({"candidate_id": candidate["candidate_id"], "reasons": reasons})
            continue

        length_score = 1.0 / length_ratio
        angle_score = 1.0 - angle_error / max(1e-9, max_opposite_angle_error)
        visual_score = _visual_score(candidate["strip_a"], candidate["strip_b"])
        siamese_score = siamese_scorer.score(candidate["strip_a"], candidate["strip_b"]) if siamese_scorer else None
        learned_score = siamese_score if siamese_score is not None else visual_score
        score = 0.25 * length_score + 0.20 * angle_score + 0.55 * learned_score
        if score < min_score:
            rejected.append({"candidate_id": candidate["candidate_id"], "reasons": ["score"]})
            continue

        record = dict(candidate)
        record.update(
            {
                "length_score": length_score,
                "opposite_angle_error": angle_error,
                "angle_score": angle_score,
                "visual_score": visual_score,
                "siamese_score": siamese_score,
                "score": score,
                "relative_rotation_degrees": (angle_a + 180.0 - angle_b) % 360.0,
                "translation": [
                    float(candidate["edge_a_geometry"]["midpoint"][0] - candidate["edge_b_geometry"]["midpoint"][0]),
                    float(candidate["edge_a_geometry"]["midpoint"][1] - candidate["edge_b_geometry"]["midpoint"][1]),
                ],
            }
        )
        filtered.append(record)

    filtered.sort(key=lambda item: item["score"], reverse=True)
    result = {
        "method": "siamese_transfer_score_v1" if siamese_scorer else "edge_profile_geometry_score_v1",
        "input_candidate_count": len(data.get("candidates", [])),
        "filtered_count": len(filtered),
        "rejected_count": len(rejected),
        "max_length_ratio": max_length_ratio,
        "max_opposite_angle_error": max_opposite_angle_error,
        "min_score": min_score,
        "pairs": filtered,
        "rejected": rejected,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
