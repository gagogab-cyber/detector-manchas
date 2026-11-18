"""Prueba de generalización: prendas de colores apagados/claros/oscuros sobre fondo blanco,
con manchas suaves. Ejecutar:  python tests/test_colores.py"""
import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.detector import detect  # noqa: E402

COLORES = {  # BGR
    "beige apagado": (150, 175, 195), "gris medio": (125, 125, 125), "gris claro": (190, 190, 190),
    "oliva apagado": (95, 120, 110), "azul marino": (70, 40, 30), "negro": (30, 30, 30),
    "rojo ladrillo": (60, 60, 150), "verde botella": (75, 100, 40), "celeste pastel": (210, 190, 160),
}


def prenda(color, seed):
    rng = np.random.default_rng(seed)
    H, W = 640, 540
    img = np.full((H, W, 3), 255, np.float32)
    cuerpo = np.array([[170, 40], [370, 40], [420, 120], [500, 190], [450, 250], [405, 215], [405, 600],
                       [135, 600], [135, 215], [90, 250], [40, 190], [120, 120]], np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [cuerpo], 1)
    tela = np.array(color, np.float32)[None, None] * np.linspace(1.08, 0.92, W)[None, :, None]   # sombreado
    tela = tela + rng.normal(0, 2.5, (H, W, 3))                                                  # tejido
    img[m > 0] = tela[m > 0]
    manchas = [(270, 240, 38), (230, 450, 30)]
    for k, (x, y, r) in enumerate(manchas):                    # manchas suaves: oscurecen y viran el tono
        blob = np.zeros((H, W), np.float32)
        cv2.circle(blob, (x, y), r, 1, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), 6)[..., None] * m[..., None]
        tono = np.array([-14, -2, 10] if k == 0 else [10, 6, -12], np.float32)
        img += blob * (tono + np.array(color, np.float32) * (-0.16 if k == 0 else 0.05))
    img = cv2.GaussianBlur(np.clip(img, 0, 255), (0, 0), 0.8)
    return np.clip(img, 0, 255).astype(np.uint8), manchas


def main():
    fallos = 0
    for i, (nombre, col) in enumerate(COLORES.items()):
        img, manchas = prenda(col, i)
        r = detect(img)
        d = r["defectos"]
        hit = [any(q["x"] <= x <= q["x"] + q["ancho"] and q["y"] <= y <= q["y"] + q["alto"] for q in d) for x, y, _ in manchas]
        extra = len(d) - sum(hit)
        ok = all(hit) and extra <= 1 and r["fondo_separado"]
        fallos += not ok
        print(f"{'OK ' if ok else 'FALLA'} {nombre:15s} manchas {sum(hit)}/2  extra {extra}  fondo separado {r['fondo_separado']}")
    print("\nTodo bien." if not fallos else f"\n{fallos} color(es) fallaron.")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
