from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torchvision import models, transforms
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
    models = None
    transforms = None


if nn is not None:
    class SiameseEdgeNetwork(nn.Module):
        def __init__(self, embedding_dim: int = 128, pretrained: bool = True) -> None:
            super().__init__()
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.projection = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, embedding_dim),
            )

        def encode(self, image: Any) -> Any:
            embedding = self.projection(self.backbone(image))
            return F.normalize(embedding, p=2, dim=1)

        def forward(self, image_a: Any, image_b: Any) -> Any:
            return self.encode(image_a), self.encode(image_b)
else:
    class SiameseEdgeNetwork:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Instala torch y torchvision para usar la red siamesa")


def build_edge_transform(image_size: int = 224) -> Any:
    if transforms is None:
        raise ImportError("Instala torchvision para preprocesar franjas")
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_siamese_model(model_path: Path, device: str = "cpu", embedding_dim: int = 128) -> SiameseEdgeNetwork:
    if torch is None:
        raise ImportError("Instala torch y torchvision para cargar la red siamesa")
    model = SiameseEdgeNetwork(embedding_dim=embedding_dim, pretrained=False)
    checkpoint = torch.load(str(model_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def contrastive_loss(embedding_a: Any, embedding_b: Any, label: Any, margin: float = 1.0) -> Any:
    if F is None:
        raise ImportError("Instala torch para calcular contrastive loss")
    distance = F.pairwise_distance(embedding_a, embedding_b)
    positive = label * distance.pow(2)
    negative = (1.0 - label) * F.relu(margin - distance).pow(2)
    return (positive + negative).mean()
