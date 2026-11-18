"""Pruebas de sombras y manchas de bajo contraste. Ejecutar:  python tests/test_sombras.py
Una mancha que solo oscurece la tela puede tener el borde difuso; antes el filtro de
sombras la descartaba. Estas pruebas comprueban que se detecte y que una sombra amplia
(o una tela limpia) siga sin marcarse.
"""
import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.detector import detect  # noqa: E402


def foto(amp=-18, blur=10, radio=28, sombra=-25, semilla=1):
    """Tela texturada con luz desigual, una sombra ancha y una mancha opcional."""
    h, w = 480, 640
    rng = np.random.default_rng(semilla)
    lab = np.zeros((h, w, 3), np.float32)
    lab[..., 0] = 55 + rng.normal(0, 0.8, (h, w))
    lab[..., 1] = rng.normal(0, 1.0, (h, w))
    lab[..., 2] = rng.normal(0, 1.0, (h, w))
    lab[..., 0] += np.linspace(-4, 4, w)[None, :]                  # luz desigual

    sh = np.zeros((h, w), np.float32)
    cv2.ellipse(sh, (470, 330), (120, 55), -20, 0, 360, 1, -1)
    lab[..., 0] += sombra * cv2.GaussianBlur(sh, (0, 0), 30)

    if amp:
        st = np.zeros((h, w), np.float32)
        cv2.circle(st, (200, 200), radio, 1, -1)
        lab[..., 0] += amp * cv2.GaussianBlur(st, (0, 0), blur)
    return (cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255).clip(0, 255).astype(np.uint8)


def cubre(defectos, cx=200, cy=200):
    return any(d["x"] <= cx <= d["x"] + d["ancho"] and d["y"] <= cy <= d["y"] + d["alto"]
               for d in defectos)


def main():
    fallos = 0
    casos = [
        ("mancha oscura de borde suave", dict(amp=-18, blur=10), 1),
        ("mancha muy difusa", dict(amp=-18, blur=18), 1),
        ("mancha pequeña", dict(amp=-18, blur=8, radio=18), 1),
        ("mancha tenue (dE ~ 4.4)", dict(amp=-5, blur=4, radio=35), 1),
        ("solo sombra", dict(amp=0), 0),
        ("tela limpia", dict(amp=0, sombra=0), 0),
    ]
    for nombre, kw, esperados in casos:
        d = detect(foto(**kw))["defectos"]
        ok = len(d) == esperados and (not esperados or cubre(d))
        fallos += not ok
        print(f"{'OK ' if ok else 'FALLA'} {nombre}: {len(d)} detectadas (esperadas {esperados})")
    print("\nTodo bien." if not fallos else f"\n{fallos} prueba(s) fallaron.")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
