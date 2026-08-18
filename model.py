"""BC-модель (CNN + MLP) и RL-политика (MLP) для PandaPickCube."""

import torch
import torch.nn as nn
import timm


class BCPolicy(nn.Module):
    """Визуомоторная BC-политика: RGB-изображение → вектор действия.

    CNN-энкодер (ResNet-18 из timm) извлекает фичи из кадра,
    MLP-декодер предсказывает 8-мерный вектор действия.
    """

    def __init__(self, action_dim: int = 8, encoder_name: str = "resnet18"):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0)
        feature_dim = self.encoder.num_features

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: RGB-изображения (B,C,H,W), float32 в диапазоне [0,1]."""
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        if obs.ndim != 4 or obs.shape[1] != 3:
            raise ValueError(
                "Ожидалось RGB-изображение в формате (B,3,H,W), "
                f"получена форма {tuple(obs.shape)}"
            )
        if obs.dtype != torch.float32:
            raise TypeError(
                "Ожидался тензор float32 в диапазоне [0,1], "
                f"получен тип {obs.dtype}"
            )
        features = self.encoder(obs)
        return self.decoder(features)


class RLPolicy(nn.Module):
    """MLP-политика для privileged state → action."""

    def __init__(self, state_dim: int = 29, action_dim: int = 8,
                 hidden_dims: tuple = (512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ELU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
