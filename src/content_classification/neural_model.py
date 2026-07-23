from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    from torchvision import models, transforms
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    models = None
    transforms = None


if nn is not None:
    class ContentClassifierNetwork(nn.Module):
        def __init__(self, num_classes: int, pretrained: bool = True) -> None:
            super().__init__()
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
            self.backbone = backbone

        def forward(self, image: Any) -> Any:
            return self.backbone(image)
else:
    class ContentClassifierNetwork:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Instala torch y torchvision para usar el clasificador de contenido")


def build_content_transform(image_size: int = 224) -> Any:
    if transforms is None:
        raise ImportError("Instala torchvision para preprocesar imagenes de contenido")
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_content_classifier(model_path: Path, device: str = "cpu") -> tuple[ContentClassifierNetwork, list[str], int]:
    if torch is None:
        raise ImportError("Instala torch y torchvision para cargar el clasificador de contenido")
    checkpoint = torch.load(str(model_path), map_location=device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))
    model = ContentClassifierNetwork(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, image_size
