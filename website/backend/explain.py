import numpy as np
import torch
import torch.nn as nn


def _backbone_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    return model.layer4(model.layer3(model.layer2(model.layer1(x))))


def classify_with_cam(model: nn.Module, tensor: torch.Tensor) -> tuple[float, np.ndarray]:
    """Return (probability of REAL, 7x7 heat map in 0..1 for the predicted class).

    The backbone runs once without gradients; only the small classifier head is
    differentiated, so this costs almost nothing beyond a normal prediction. The
    pre-sigmoid logit is used so saturated scores still give a usable gradient.
    """
    with torch.no_grad():
        features = _backbone_features(model, tensor)
    features = features.detach().requires_grad_(True)

    with torch.enable_grad():
        logit = model.fc[:-1](torch.flatten(model.avgpool(features), 1)).sum()
        probability = torch.sigmoid(logit).item()
        score = logit if probability > 0.5 else -logit
        (grad,) = torch.autograd.grad(score, features)

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * features.detach()).sum(dim=1))[0]
    peak = cam.max()
    if peak > 0:
        cam = cam / peak
    return probability, cam.numpy()


def edge_share(cam: np.ndarray) -> float:
    """Fraction of heat in the outer ring of cells (uniform heat would be about 0.49 for 7x7)."""
    total = float(cam.sum())
    if total == 0:
        return 0.0
    return 1.0 - float(cam[1:-1, 1:-1].sum()) / total
