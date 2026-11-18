"""Conversión a CIELAB y diferencia de color (ΔE)."""
import cv2
import numpy as np
from skimage.color import deltaE_ciede2000


def bgr_to_lab(bgr_u8):
    """BGR uint8 -> Lab float32 (L: 0-100, a/b: aprox -127..127)."""
    return cv2.cvtColor(bgr_u8.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def delta_e(lab1, lab2, kL=1.0):
    """CIEDE2000. kL=1 pesa igual la luminosidad y el color: así se ven manchas oscuras/mojadas.
    (kL=2 es el valor textil clásico, pero apaga demasiado las manchas grises.)"""
    lab2 = np.broadcast_to(lab2, lab1.shape)
    return deltaE_ciede2000(lab1, lab2, kL=kL).astype(np.float32)
