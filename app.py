"""Servidor web (Flask).  Ejecutar:  python app.py   y abrir  http://localhost:5000
El motor de detección es el mismo de src/ (misma precisión que la versión de escritorio)."""
import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(100_000_000))   # protege la RAM (4 GB)

import base64
import socket
import time
import cv2
from flask import Flask, jsonify, render_template, request
from src.cnn import load_cnn
from src.detector import Params, detect
from src.preprocess import crop_pct, load_bgr

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024        # fotos de hasta 20 MB
_cnn = load_cnn()                                           # None si no hay models/stain_cnn.pt


def _num(key, default, lo, hi):
    try:
        v = float(request.form.get(key, default))
    except (TypeError, ValueError):
        v = default
    return min(max(v, lo), hi)


def _flag(key, default):
    v = request.form.get(key)
    return default if v is None else v.lower() in ("1", "true", "on")


def _data_url(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/config")
def config():
    return jsonify(cnn=_cnn is not None)


@app.get("/api/salud")
def salud():
    return jsonify(estado="ok")


@app.post("/api/analizar")
def analizar():
    f = request.files.get("imagen")
    if f is None:
        return jsonify(error="Falta la imagen."), 400
    max_side = int(_num("max_side", 640, 320, 1024))
    try:
        bgr = load_bgr(f.read(), max_side)
    except ValueError:
        return jsonify(error="No se pudo leer la imagen. Usa JPG o PNG."), 400
    bgr = crop_pct(bgr, _num("crop_izq", 0, 0, 45), _num("crop_der", 0, 0, 45),
                   _num("crop_arr", 0, 0, 45), _num("crop_aba", 0, 0, 45))
    p = Params(tol=_num("tol", 3.0, 1.5, 10), z_thr=_num("z_thr", 4.0, 3, 8),
               min_area_pct=_num("min_area", 0.05, 0.01, 2), max_side=max_side,
               boost_dark=_flag("boost", True), filter_shadows=_flag("sombras", True),
               filter_prints=_flag("estampados", True), ignore_skin=_flag("piel", True),
               use_palette=_flag("paleta", False), isolate_garment=_flag("fondo", True))
    t0 = time.perf_counter()
    res = detect(bgr, p, _cnn if _flag("cnn", False) else None)
    ms = int((time.perf_counter() - t0) * 1000)
    d = res["defectos"]
    return jsonify(
        veredicto="FUERA DE TOLERANCIA" if d else "APROBADA",
        total=len(d), area_danada_pct=res["area_danada_pct"], fondo_separado=res["fondo_separado"], tiempo_ms=ms,
        ancho=int(bgr.shape[1]), alto=int(bgr.shape[0]), defectos=d,
        original=_data_url(bgr), marcada=_data_url(res["marcada"]), heatmap=_data_url(res["heatmap"]))


@app.errorhandler(413)
def demasiado_grande(_):
    return jsonify(error="La imagen supera los 20 MB."), 413


def _ip_local():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Inspección de tela lista:\n    En esta PC:   http://localhost:{port}\n"
          f"    En tu celular (misma Wi-Fi):  http://{_ip_local()}:{port}\n")
    serve(app, host="0.0.0.0", port=port, threads=2)     # 2 hilos = RAM acotada
