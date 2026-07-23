from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.compatibility.siamese_model import SiameseEdgeNetwork, build_edge_transform, contrastive_loss, torch


class EdgePairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_path: Path, image_size: int = 224) -> None:
        if torch is None:
            raise ImportError("torch no esta instalado")
        data = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
        raw_pairs = data.get("pairs", data.get("candidates", []))
        self.pairs = [pair for pair in raw_pairs if "label" in pair or "is_match" in pair]
        if not self.pairs:
            raise ValueError("No hay pares etiquetados. Cada par debe tener 'label' o 'is_match'.")
        self.transform = build_edge_transform(image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_rgb(self, path: str) -> Any:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise FileNotFoundError(f"No se pudo leer franja: {path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            rgb = image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
            image = rgb.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        return self.transform(image)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        pair = self.pairs[index]
        image_a = self._read_rgb(pair["strip_a"])
        image_b = self._read_rgb(pair["strip_b"])
        label_value = pair.get("label", pair.get("is_match"))
        label = torch.tensor(float(label_value), dtype=torch.float32)
        return image_a, image_b, label


def train_siamese(
    pairs_path: Path,
    output_path: Path,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    embedding_dim: int = 128,
    image_size: int = 224,
    device: str | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise ImportError("Instala torch y torchvision para entrenar la red siamesa")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EdgePairDataset(pairs_path, image_size=image_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SiameseEdgeNetwork(embedding_dim=embedding_dim, pretrained=True).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for image_a, image_b, labels in loader:
            image_a = image_a.to(selected_device)
            image_b = image_b.to(selected_device)
            labels = labels.to(selected_device)
            optimizer.zero_grad()
            emb_a, emb_b = model(image_a, image_b)
            loss = contrastive_loss(emb_a, emb_b, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * image_a.size(0)
        history.append({"epoch": epoch + 1, "loss": total_loss / len(dataset)})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": embedding_dim,
            "image_size": image_size,
            "history": history,
        },
        str(output_path),
    )
    return {"output_path": str(output_path.as_posix()), "samples": len(dataset), "history": history}


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena red siamesa de compatibilidad de bordes")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/siamese_model.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    result = train_siamese(args.pairs, args.output, args.epochs, args.batch_size, args.learning_rate)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
