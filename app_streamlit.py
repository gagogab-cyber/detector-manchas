"""Interfaz Streamlit. Ejecutar:  streamlit run streamlit_app.py
En Streamlit Cloud el archivo principal es streamlit_app.py (no el botón play de VS Code)."""
import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(100_000_000))  # protege la RAM en la nube

import cv2
import pandas as pd
import streamlit as st
from src.cnn import RUTA_MODELO, load_cnn
from src.detector import detect, Params
from src.preprocess import load_bgr, crop_pct

st.set_page_config(page_title="Inspección de tela", layout="wide")
st.title("🧵 Detección de manchas y defectos de color en tela")


@st.cache_resource
def get_cnn():
    return load_cnn()


with st.sidebar:
    st.header("Ajustes")
    st.caption("Los valores por defecto sirven para la mayoría de las fotos.")
    tol = st.slider("Tolerancia ΔE", 1.5, 10.0, 3.0, 0.5,
                    help="Una zona es defecto si su diferencia de color (ΔE) promedio supera esto")
    z_thr = st.slider("Sensibilidad (más bajo = detecta más)", 3.0, 8.0, 4.0, 0.5,
                      help="Qué tan 'rara' debe ser una zona frente al resto de la tela")
    min_area = st.slider("Área mínima (%)", 0.01, 2.0, 0.05, 0.01,
                         help="Si no detecta defectos chicos, baja esto. Si hay mucho puntito, súbelo")
    max_side = st.select_slider("Resolución de trabajo (px)", [480, 640, 800, 1024], 640)
    boost = st.checkbox("Reforzar manchas claras en tela oscura", True)
    shadows = st.checkbox("Filtrar sombras y arrugas", True)
    prints = st.checkbox("Filtrar estampados repetidos", True)
    skin = st.checkbox("Ignorar piel (manos, brazos)", True)
    fondo = st.checkbox("Separar la prenda de un fondo liso (fotos de catálogo)", True)
    palette = st.checkbox("Prenda con estampado grande (usar paleta)", False,
                          help="Solo para bloques grandes de color. Puede esconder manchas, déjalo apagado en telas lisas")
    use_cnn = st.checkbox("Usar CNN ligera", False) if os.path.exists(RUTA_MODELO) else False
    st.subheader("Recortar bordes (%)")
    st.caption("Quita manos, mangas, pared u otras cosas que no sean la tela.")
    cl = st.slider("Izquierda", 0, 45, 0)
    cr = st.slider("Derecha", 0, 45, 0)
    ct = st.slider("Arriba", 0, 45, 0)
    cb = st.slider("Abajo", 0, 45, 0)

use_cam = st.toggle("Usar cámara")
file = st.camera_input("Foto de la prenda") if use_cam else \
    st.file_uploader("Sube la foto de la prenda", type=["jpg", "jpeg", "png"])

if file:
    if len(file.getvalue()) > 20 * 1024 * 1024:
        st.error("La imagen supera los 20 MB. Sube una foto más pequeña.")
        st.stop()
    bgr = load_bgr(file.getvalue(), max_side)
    bgr = crop_pct(bgr, cl, cr, ct, cb)
    p = Params(tol=tol, z_thr=z_thr, min_area_pct=min_area, boost_dark=boost, filter_shadows=shadows,
               filter_prints=prints, ignore_skin=skin, isolate_garment=fondo, use_palette=palette, max_side=max_side)
    with st.spinner("Analizando..."):
        res = detect(bgr, p, get_cnn() if use_cnn else None)

    c1, c2, c3 = st.columns(3)
    c1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Zona analizada", use_container_width=True)
    c2.image(cv2.cvtColor(res["marcada"], cv2.COLOR_BGR2RGB), caption="Zonas fuera de tolerancia", use_container_width=True)
    c3.image(cv2.cvtColor(res["heatmap"], cv2.COLOR_BGR2RGB), caption="Mapa de anomalía (rojo = más raro)", use_container_width=True)

    d = res["defectos"]
    st.metric("Defectos encontrados", len(d))
    if d:
        st.error("❌ Prenda con zonas fuera de tolerancia")
        df = pd.DataFrame(d)
        st.dataframe(df, use_container_width=True)
        st.download_button("Descargar tabla (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "defectos.csv", "text/csv")
    else:
        st.success("✅ Prenda dentro de tolerancia")
    ok, png = cv2.imencode(".png", res["marcada"])
    st.download_button("Descargar imagen marcada", png.tobytes(), "resultado.png", "image/png")

    with st.expander("¿Cómo leer el resultado?"):
        st.markdown(
            "- **Recuadro rojo** = defecto grave (ΔE medio > 2 × tolerancia). **Naranja** = leve.\n"
            "- **Línea amarilla** = contorno exacto de la zona detectada.\n"
            "- **ΔE** = diferencia de color (CIEDE2000) de la zona contra el color normal de la tela.\n"
            "- **Mapa de anomalía** = qué tan rara es cada parte; los recuadros salen de ahí.\n"
            "- Si no detecta algo visible, baja la **Sensibilidad** o el **Área mínima**. "
            "Si marca mucho, súbelos o recorta los bordes.")
