"""Pruebas de la web. Ejecutar:  python tests/test_web.py
Comprueba que la API devuelve EXACTAMENTE lo mismo que el motor directo (misma precisión),
y que maneja bien errores y entradas raras."""
import io
import os
import sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app import app  # noqa: E402
from src.detector import detect, Params  # noqa: E402
from src.preprocess import load_bgr  # noqa: E402

EJEMPLOS = ["camisa_gris_mancha_oscura", "tela_clara_manchas_amarillas",
            "tela_negra_manchas_naranjas", "tela_gris_roturas", "polo_verde_fondo_blanco"]
fallos = 0


def check(cond, msg):
    global fallos
    fallos += not cond
    print(("OK    " if cond else "FALLA ") + msg)


c = app.test_client()
check(c.get("/").status_code == 200, "GET / carga la página")
check(c.get("/static/js/main.js").status_code == 200 and c.get("/static/css/style.css").status_code == 200, "archivos estáticos")
check(c.get("/api/salud").get_json()["estado"] == "ok", "GET /api/salud")

for nombre in EJEMPLOS:
    ruta = os.path.join(ROOT, "ejemplos", nombre + ".png")
    raw = open(ruta, "rb").read()
    r = c.post("/api/analizar", data={"imagen": (io.BytesIO(raw), nombre + ".png")}, content_type="multipart/form-data")
    j = r.get_json()
    directo = detect(load_bgr(raw, 640), Params())
    mismos = [(d["x"], d["y"], d["ancho"], d["alto"], d["ΔE medio"]) for d in j["defectos"]] == \
             [(d["x"], d["y"], d["ancho"], d["alto"], d["ΔE medio"]) for d in directo["defectos"]]
    check(r.status_code == 200 and mismos and j["area_danada_pct"] == directo["area_danada_pct"] and j["fondo_separado"] == directo["fondo_separado"],
          f"{nombre}: web == motor directo ({j['total']} defectos, {j['area_danada_pct']}% área, {j['tiempo_ms']} ms)")
    check(j["marcada"].startswith("data:image/jpeg;base64,") and j["veredicto"] == "FUERA DE TOLERANCIA", f"{nombre}: imágenes y veredicto")

# parámetros de la interfaz llegan bien (y valores absurdos se limitan, no rompen)
raw = open(os.path.join(ROOT, "ejemplos", "tela_clara_manchas_amarillas.png"), "rb").read()
r = c.post("/api/analizar", data={"imagen": (io.BytesIO(raw), "a.png"), "tol": "9999", "z_thr": "abc",
           "crop_izq": "999", "max_side": "100000", "boost": "0"}, content_type="multipart/form-data")
check(r.status_code == 200, "parámetros fuera de rango o inválidos se limitan sin romper")

check(c.post("/api/analizar", data={}, content_type="multipart/form-data").status_code == 400, "sin imagen -> 400")
r = c.post("/api/analizar", data={"imagen": (io.BytesIO(b"esto no es una imagen"), "x.png")}, content_type="multipart/form-data")
check(r.status_code == 400 and "error" in r.get_json(), "archivo corrupto -> 400 con mensaje")
app.config["MAX_CONTENT_LENGTH"] = 1000
r = c.post("/api/analizar", data={"imagen": (io.BytesIO(b"0" * 5000), "x.png")}, content_type="multipart/form-data")
check(r.status_code == 413, "archivo demasiado grande -> 413")

print("\nTodo bien." if not fallos else f"\n{fallos} prueba(s) fallaron.")
sys.exit(1 if fallos else 0)
