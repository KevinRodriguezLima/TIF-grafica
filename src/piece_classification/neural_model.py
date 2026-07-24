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

PIECE_CLASSES = ["corner", "border", "interior"]


if nn is not None:
    class PieceClassifierNetwork(nn.Module):
        def __init__(self, num_classes: int = 3, pretrained: bool = True) -> None:
            super().__init__()
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                4,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                backbone.conv1.weight[:, :3] = old_conv.weight
                backbone.conv1.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
            backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
            self.backbone = backbone

        def forward(self, image: Any) -> Any:
            return self.backbone(image)
else:
    class PieceClassifierNetwork:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Instala torch y torchvision para usar el clasificador de piezas")


def build_piece_transform(image_size: int = 224) -> Any:
    if transforms is None:
        raise ImportError("Instala torchvision para preprocesar piezas")
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]),
        ]
    )


def load_piece_classifier(model_path: Path, device: str = "cpu") -> tuple[PieceClassifierNetwork, list[str], int]:
    if torch is None:
        raise ImportError("Instala torch y torchvision para cargar el clasificador de piezas")
    checkpoint = torch.load(str(model_path), map_location=device)
    class_names = checkpoint.get("class_names", PIECE_CLASSES)
    image_size = int(checkpoint.get("image_size", 224))
    model = PieceClassifierNetwork(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device)
    model.eval()
    return model, class_names, image_size
