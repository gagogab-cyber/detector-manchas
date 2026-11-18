"""Pipeline de detección de manchas/defectos en tela.

Resumen (cada paso resuelve un problema real que apareció al probar fotos):
  1. Preproceso: quita ruido y la textura del tejido (blur relativo al tamaño de la foto).
  2. CIELAB: separa luminosidad (L*) de color (a*, b*).
  3. "Color normal" de la tela con un ajuste de superficie ROBUSTO (las manchas grandes no lo contaminan).
  4. Puntaje de anomalía: luminosidad y color por separado, cada uno normalizado por el
     ruido de ESA tela. Se queda el canal que mejor separa la mancha en cada foto.
  5. Umbral con histéresis por zona (la mancha crece hasta su borde real).
  6. Filtros espaciales: sombras/arrugas (por borde y contraste local, no solo por brillo),
     estampados repetidos, piel grande, fragmentos de borde.
  7. ΔE CIEDE2000 por zona (lo que se reporta) y salida visual.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from .color import bgr_to_lab, delta_e
from .garment import garment_mask
from .preprocess import denoise, background, flatten_illumination


@dataclass
class Params:
    tol: float = 3.0              # una zona es defecto si su ΔE medio >= tol
    z_thr: float = 4.0            # sensibilidad: más bajo = detecta más
    min_area_pct: float = 0.05    # área mínima de una zona (% de la foto)
    boost_dark: bool = True       # más sensible a manchas claras sobre tela oscura
    filter_shadows: bool = True   # descarta sombras y arrugas
    filter_prints: bool = True    # descarta regiones repetidas (estampado de motivos)
    repeat_min: int = 3           # >= 3 regiones parecidas = patrón, no mancha
    ignore_skin: bool = True      # ignora zonas GRANDES de piel (manos, brazos)
    isolate_garment: bool = True  # con fondo liso (foto de catálogo) analiza solo la prenda
    use_palette: bool = False     # solo para estampados de bloques grandes
    palette_share: float = 0.05
    max_side: int = 640           # lado máximo de trabajo (lo usa la carga de la imagen)


# ---------------------------------------------------------------- 1) mapa de anomalía
def _zscore(x, floor, mask=None):
    """Desviaciones respecto al ruido de la tela. El ruido se mide solo con la mitad baja
    de los valores (tela sin mancha), así las manchas grandes no inflan el umbral.
    Con `mask`, las estadísticas salen solo de la prenda (no del fondo)."""
    v = x if mask is None else x[mask]
    med = np.median(v)
    low = v[v <= med]
    sig = max(1.4826 * np.median(np.abs(low - med)), floor)
    return (x - med) / sig


def _palette(lab, share, k=5):
    """Colores 'legítimos' (base + bloques de estampado) con k-means."""
    cv2.setRNGSeed(0)
    step = max(1, lab.shape[0] * lab.shape[1] // 20000)
    pts = np.ascontiguousarray(lab.reshape(-1, 3)[::step])
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    frac = np.bincount(labels.ravel(), minlength=k) / len(pts)
    keep = centers[frac >= share]
    return keep if len(keep) else centers[[frac.argmax()]]


def _big_skin_mask(bgr):
    """Solo zonas GRANDES de color piel (>3% de la foto). Una mancha chica naranja o
    rosada tiene color 'piel' pero no es piel."""
    H, W = bgr.shape[:2]
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    skin = (cv2.inRange(ycc, (0, 135, 80), (255, 175, 125)) > 0).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.03 * H * W]
    return cv2.dilate(np.isin(lbl, big).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0


def anomaly_map(bgr, p):
    H, W = bgr.shape[:2]
    lab = bgr_to_lab(denoise(bgr))
    sigma = float(np.clip(0.007 * max(H, W), 1.5, 4.0))     # relativo al tamaño de la foto
    lab = cv2.GaussianBlur(lab, (0, 0), sigma)               # apaga la textura del tejido
    gm = garment_mask(lab) if p.isolate_garment else None    # prenda sobre fondo liso (o None)
    lab = flatten_illumination(lab, gm)                      # apaga degradados de luz

    bgLmap = background(lab[..., 0], mask=gm)
    bgL = float(np.median(bgLmap if gm is None else bgLmap[gm]))
    bg = np.dstack([np.full(lab.shape[:2], bgL, np.float32),
                    background(lab[..., 1], mask=gm), background(lab[..., 2], mask=gm)])
    de = delta_e(lab, bg)                                    # ΔE CIEDE2000 (se reporta)
    dL = lab[..., 0] - bg[..., 0]
    dC = np.hypot(lab[..., 1] - bg[..., 1], lab[..., 2] - bg[..., 2])

    wL = np.where(dL > 0, 1.4, 1.0) if (p.boost_dark and bgL < 40) else 1.0
    score = np.maximum(_zscore(np.abs(dL), 0.8, gm) * wL, _zscore(dC, 0.4, gm)).astype(np.float32)
    if gm is not None:
        score[~gm] = 0                                       # el fondo no se analiza
    score[de < p.tol * 0.6] = 0                              # diferencias mínimas no cuentan

    if p.use_palette:
        de_pal = np.full(lab.shape[:2], 1e3, np.float32)
        for c in _palette(lab, p.palette_share):
            de_pal = np.minimum(de_pal, delta_e(lab, c.astype(np.float32)))
        score[de_pal < p.tol] = 0
    if p.ignore_skin:
        skin = _big_skin_mask(bgr)
        ref = int(gm.sum()) if gm is not None else score.size
        # Manos/brazos son una parte menor. Si el "color piel" domina (>35%), es el color de la
        # prenda (beige, salmón, rojo ladrillo...) y NO se ignora.
        skin_n = int((skin & gm).sum()) if gm is not None else int(skin.sum())
        if skin_n <= 0.35 * ref:
            score[skin] = 0
    return lab, bg, de, dL, score, gm


# ---------------------------------------------------------------- 2) segmentación
def segment(score, thr):
    """Histéresis por zona: semilla fuerte, crece hasta una fracción de SU intensidad."""
    strong = score > thr
    n, lw = cv2.connectedComponents((score > thr * 0.6).astype(np.uint8), connectivity=8)
    mask = np.zeros(score.shape, np.uint8)
    for i in np.unique(lw[strong]):
        if i == 0:
            continue
        comp = lw == i
        core = np.median(score[comp & strong])
        mask[comp & (score > max(thr * 0.6, 0.5 * core))] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, cnt, -1, 1, -1)                   # rellena huecos internos
    return mask


# ---------------------------------------------------------------- 3) análisis de zonas
def analyze_regions(bgr, mask, lab, bg, de, dL, p, cnn=None, gm=None):
    H, W = mask.shape
    L = lab[..., 0]
    grad = cv2.magnitude(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8,
                         cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = p.min_area_pct / 100 * H * W
    k3 = np.ones((3, 3), np.uint8)
    # píxeles pegados al exterior de la prenda: contorno, cuello y puños (ribetes de diseño)
    near_out = (cv2.dilate((~gm).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) if gm is not None else None
    # Anillo alrededor de cada candidato: permite distinguir una mancha local de un degradado
    # de luz. Se calcula una sola vez para toda la imagen.
    if p.filter_shadows:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        ring_zone = cv2.dilate(mask, kernel) > 0
    else:
        ring_zone = None
    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if max(w, h) / max(1, min(w, h)) > 5:
            continue                                         # muy alargado: arruga o costura
        sl = (slice(y, y + h), slice(x, x + w))
        sub = lbl[sl] == i
        de_mean = float(de[sl][sub].mean())
        if de_mean < p.tol:
            continue                                         # dentro de tolerancia
        if near_out is not None and near_out[sl][sub].any():
            continue                                         # toca el borde de la prenda: ribete/costura, no mancha
        dl_vals = dL[sl][sub]
        dl = float(dl_vals.mean())
        amp = float(np.abs(dl_vals).mean())                  # amplitudes con signos opuestos no se cancelan
        dab = float(np.hypot(lab[sl][..., 1] - bg[sl][..., 1], lab[sl][..., 2] - bg[sl][..., 2])[sub].mean())
        border = sub & ~cv2.erode(sub.astype(np.uint8), k3).astype(bool)
        edge = float(grad[sl][border].mean()) if border.any() else 0.0
        extent = area / (w * h)
        # Sombra/arruga: solo cambia la luminosidad. Se mantiene el filtro clásico para zonas
        # alargadas o de borde suave y débil, pero un borde definido o una zona compacta con
        # suficiente ΔE y contraste local se acepta (antes se perdían manchas oscuras difusas).
        if p.filter_shadows and dab < 3:
            if extent < 0.45:
                continue
            if edge < 1.0:
                transition = edge / max(amp, 1.0)
                around = ring_zone[sl] & (lbl[sl] == 0)
                ring_dl = float(dL[sl][around].mean()) if around.any() else dl
                local_contrast = float(np.abs(dl_vals - ring_dl).mean()) / max(amp, 1.0)
                # Las zonas muy pequeñas y de borde difuso son indistinguibles del tejido:
                # se exige algo más de tamaño antes de aceptarlas con poca evidencia.
                if (amp < 4 or de_mean < p.tol + 1.0 or 100 * area / (H * W) < 0.2
                        or transition < 0.03 or local_contrast < 0.62):
                    continue
        # Pedacitos pegados al borde de la foto: artefactos
        if (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1) and 100 * area / (H * W) < 0.5:
            continue
        if cnn is not None and cnn(bgr[sl]) < 0.5:
            continue
        cands.append({"box": (x, y, w, h), "sub": sub, "area": int(area), "dl": dl, "dab": dab,
                      "color": lab[sl][sub].mean(0)})

    # Estampado = muchas regiones parecidas (mismo color y tamaño). Una mancha suele ser única.
    if p.filter_prints and len(cands) > p.repeat_min:
        cands = [a for a in cands if sum(
            1 for b in cands if b is not a
            and np.linalg.norm((a["color"] - b["color"]) * [0.5, 1, 1]) < 6
            and 0.5 <= b["area"] / a["area"] <= 2) < p.repeat_min]

    final = np.zeros((H, W), np.uint8)
    defects = []
    for c in cands:
        x, y, w, h = c["box"]
        final[y:y + h, x:x + w][c["sub"]] = 255
        vals = de[y:y + h, x:x + w][c["sub"]]
        if c["dab"] > abs(c["dl"]):
            tipo = "cambio de color"
        else:
            tipo = "más clara" if c["dl"] > 0 else "más oscura"
        defects.append({"x": int(x), "y": int(y), "ancho": int(w), "alto": int(h),
                        "area_%": round(100 * c["area"] / (H * W), 2),
                        "ΔE medio": round(float(vals.mean()), 1),
                        "ΔE máx": round(float(vals.max()), 1),
                        "tipo": tipo,
                        "nivel": "grave" if vals.mean() > 2 * p.tol else "leve"})
    return final, defects


# ---------------------------------------------------------------- 4) salida visual
def draw(bgr, final, defects, score, thr):
    heat = cv2.applyColorMap(np.clip(score / (2 * thr) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.addWeighted(bgr, 0.5, heat, 0.5, 0)
    marked = bgr.copy()
    cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked, cnts, -1, (0, 255, 255), 1)
    for d in defects:
        col = (0, 0, 255) if d["nivel"] == "grave" else (0, 165, 255)
        cv2.rectangle(marked, (d["x"], d["y"]), (d["x"] + d["ancho"], d["y"] + d["alto"]), col, 2)
        cv2.putText(marked, f'dE {d["ΔE medio"]}', (d["x"], max(12, d["y"] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return marked, heat


def detect(bgr, p=Params(), cnn=None):
    lab, bg, de, dL, score, gm = anomaly_map(bgr, p)
    mask = segment(score, p.z_thr)
    final, defects = analyze_regions(bgr, mask, lab, bg, de, dL, p, cnn, gm)
    marked, heat = draw(bgr, final, defects, score, p.z_thr)
    return {"marcada": marked, "heatmap": heat, "mascara": final,
            "defectos": defects, "umbral": float(p.z_thr),
            "fondo_separado": gm is not None,
            "area_danada_pct": round(100.0 * np.count_nonzero(final) / (int(gm.sum()) if gm is not None else final.size), 2)}
