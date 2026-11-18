"""Pruebas de regresión. Ejecutar:  python tests/test_regresion.py
Verifica, con parámetros POR DEFECTO, que cada mancha conocida quede dentro de algún recuadro
detectado y que no aparezcan demasiados falsos positivos. Cada caso cubre un problema real:
  - camisa gris:   mancha oscura/mojada (solo cambia la luminosidad) + piel/pared/sombras en la foto
  - tela clara:    manchas amarillas GRANDES sobre tela arrugada (solo cambia el color)
  - tela negra:    manchas naranjas (color 'piel') sobre tela oscura texturada
  - tela gris:     pequeñas roturas claras/oscuras
  - polo verde:    prenda sobre FONDO BLANCO con manchas tenues; cuello, puños y contorno no son defectos
  - sintética:     sombra suave + estampado de puntos + luz desigual (no deben marcarse)
"""
import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.detector import detect  # noqa: E402

# nombre -> (centros (x, y) de manchas conocidas, falsos positivos tolerados)
CASOS = {
    "camisa_gris_mancha_oscura": ([(233, 154), (195, 260)], 1),
    "tela_clara_manchas_amarillas": ([(299, 18), (206, 98), (285, 240), (178, 241)], 1),
    "tela_negra_manchas_naranjas": ([(198, 414), (251, 763), (248, 850)], 3),
    "tela_gris_roturas": ([(245, 177), (298, 310), (232, 318)], 1),
    "polo_verde_fondo_blanco": ([(396, 146), (360, 367), (153, 489)], 1),
}


def cubre(defectos, cx, cy):
    return any(d["x"] <= cx <= d["x"] + d["ancho"] and d["y"] <= cy <= d["y"] + d["alto"] for d in defectos)


def sintetica():
    rng = np.random.default_rng(0)
    img = np.full((480, 640, 3), (40, 30, 30), np.float32) + rng.normal(0, 4, (480, 640, 3))
    img *= np.linspace(0.8, 1.15, 640)[None, :, None]
    for x in range(40, 600, 70):
        for y in range(40, 440, 70):
            cv2.circle(img, (x, y), 8, (120, 90, 60), -1)
    sh = np.zeros((480, 640), np.float32)
    cv2.ellipse(sh, (450, 330), (80, 40), 0, 0, 360, 1, -1)
    img *= (1 - 0.35 * cv2.GaussianBlur(sh, (0, 0), 20))[..., None]
    cv2.ellipse(img, (200, 200), (30, 18), 20, 0, 360, (95, 85, 80), -1)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 0.8), 0, 255).astype(np.uint8)


def main():
    fallos = 0
    for nombre, (centros, fp_max) in CASOS.items():
        r = detect(cv2.imread(os.path.join(ROOT, "ejemplos", nombre + ".png")))
        d = r["defectos"]
        perdidas = [c for c in centros if not cubre(d, *c)]
        extra = len(d) - (len(centros) - len(perdidas))
        ok = not perdidas and extra <= fp_max
        fallos += not ok
        print(f"{'OK ' if ok else 'FALLA'} {nombre}: {len(d)} detectadas, {len(perdidas)} manchas perdidas, {extra} extra")
    r = detect(sintetica())
    ok = len(r["defectos"]) == 1 and cubre(r["defectos"], 200, 200)
    fallos += not ok
    print(f"{'OK ' if ok else 'FALLA'} sintetica: {len(r['defectos'])} detectadas (debe ser 1: solo la mancha)")
    print("\nTodo bien." if not fallos else f"\n{fallos} prueba(s) fallaron.")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
