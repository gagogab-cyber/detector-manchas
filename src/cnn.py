"""CNN ligera OPCIONAL (~7 mil parámetros): confirma si un recorte es mancha real.
Solo se usa si existe models/stain_cnn.pt (entrénala con train_cnn.py)."""
import os
import cv2
import numpy as np

RUTA_MODELO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "stain_cnn.pt")


def build_net():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(32, 1), nn.Sigmoid())


def to_tensor(patch_bgr):
    import torch
    x = cv2.resize(patch_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1)


def load_cnn(path=RUTA_MODELO):
    """Devuelve una función patch->prob de mancha, o None si no hay modelo/torch."""
    if not os.path.exists(path):
        return None
    try:
        import torch
    except ImportError:
        return None
    net = build_net()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()

    def predict(patch_bgr):
        with torch.no_grad():
            return float(net(to_tensor(patch_bgr)[None])[0])
    return predict
