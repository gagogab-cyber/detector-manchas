"""Uso por consola (sin interfaz):  python cli.py foto.jpg [resultado.png]"""
import sys
import cv2
from src.detector import detect, Params
from src.preprocess import load_bgr

if len(sys.argv) < 2:
    sys.exit("Uso: python cli.py foto.jpg [resultado.png]")
p = Params()
with open(sys.argv[1], "rb") as f:
    bgr = load_bgr(f.read(), p.max_side)
res = detect(bgr, p)
out = sys.argv[2] if len(sys.argv) > 2 else "resultado.png"
cv2.imwrite(out, cv2.hconcat([bgr, res["marcada"], res["heatmap"]]))
print(f"{len(res['defectos'])} defecto(s) -> {out}")
for d in res["defectos"]:
    print(f"  ({d['x']},{d['y']}) {d['ancho']}x{d['alto']}  ΔE {d['ΔE medio']}  {d['tipo']}  [{d['nivel']}]")
