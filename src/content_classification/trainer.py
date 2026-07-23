from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.content_classification.neural_model import ContentClassifierNetwork, torch


def train_content_classifier(
    dataset_dir: Path,
    output_path: Path,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    image_size: int = 224,
    device: str | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise ImportError("Instala torch y torchvision para entrenar el clasificador de contenido")
    from torchvision import datasets, transforms

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(str(dataset_dir), transform=transform)
    if not dataset.samples:
        raise ValueError("El dataset de contenido no tiene imagenes")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ContentClassifierNetwork(num_classes=len(dataset.classes), pretrained=True).to(selected_device)
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
            "class_names": dataset.classes,
            "image_size": image_size,
            "history": history,
        },
        str(output_path),
    )
    return {"output_path": str(output_path.as_posix()), "samples": len(dataset), "classes": dataset.classes, "history": history}


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena clasificador del contenido reconstruido")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/content_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    result = train_content_classifier(args.dataset, args.output, args.epochs, args.batch_size, args.learning_rate)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
