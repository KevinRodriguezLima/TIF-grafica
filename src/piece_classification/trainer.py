from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.piece_classification.neural_model import PIECE_CLASSES, PieceClassifierNetwork, build_piece_transform, torch


class PieceDataset(torch.utils.data.Dataset):
    def __init__(self, features_path: Path, labels_path: Path, image_size: int = 224) -> None:
        if torch is None:
            raise ImportError("Instala torch y torchvision para entrenar el clasificador de piezas")
        features = json.loads(Path(features_path).read_text(encoding="utf-8"))
        labels_data = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        labels = labels_data.get("labels", labels_data)
        self.labels = {item["piece_id"]: item["label"] for item in labels} if isinstance(labels, list) else labels
        self.pieces = [piece for piece in features.get("pieces", []) if piece["piece_id"] in self.labels]
        if not self.pieces:
            raise ValueError("No hay piezas con etiqueta para entrenar")
        self.class_to_idx = {name: index for index, name in enumerate(PIECE_CLASSES)}
        self.transform = build_piece_transform(image_size)

    def __len__(self) -> int:
        return len(self.pieces)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        piece = self.pieces[index]
        image = cv2.imread(piece["cropped_path"], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(piece["mask_path"], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"No se pudo leer pieza o mascara: {piece['piece_id']}")
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))
        rgba = np.dstack([rgb, mask]).astype(np.uint8)
        label = self.class_to_idx[self.labels[piece["piece_id"]]]
        return self.transform(rgba), torch.tensor(label, dtype=torch.long)


def train_piece_classifier(
    features_path: Path,
    labels_path: Path,
    output_path: Path,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    image_size: int = 224,
    device: str | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise ImportError("Instala torch y torchvision para entrenar el clasificador de piezas")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PieceDataset(features_path, labels_path, image_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PieceClassifierNetwork(num_classes=len(PIECE_CLASSES), pretrained=True).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        for images, labels in loader:
            images = images.to(selected_device)
            labels = labels.to(selected_device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
        history.append({"epoch": epoch + 1, "loss": total_loss / len(dataset), "accuracy": correct / len(dataset)})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": PIECE_CLASSES,
            "image_size": image_size,
            "history": history,
        },
        str(output_path),
    )
    return {"output_path": str(output_path.as_posix()), "samples": len(dataset), "history": history}


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena clasificador de piezas corner/border/interior")
    parser.add_argument("--features", type=Path, default=Path("metadata/pieces_features.json"))
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/piece_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    result = train_piece_classifier(args.features, args.labels, args.output, args.epochs, args.batch_size, args.learning_rate)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
