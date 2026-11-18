"""Separa la prenda de un fondo liso (fotos de catálogo / mesa blanca)."""
import cv2
import numpy as np
from .color import delta_e


def garment_mask(lab):
    """Máscara booleana de la prenda, o None si la tela llena la foto (no hay fondo liso).
    Fondo = zona de color uniforme CONECTADA al borde de la foto. Así las rayas blancas de un
    cuello o puño (que están dentro de la prenda) no se confunden con fondo."""
    H, W = lab.shape[:2]
    m = max(3, int(0.03 * min(H, W)))
    strip = np.zeros((H, W), bool)
    strip[:m], strip[-m:], strip[:, :m], strip[:, -m:] = True, True, True, True
    c = np.median(lab[strip], axis=0).astype(np.float32)
    d = delta_e(lab, c)
    sel = strip & (d < 10)
    if sel.sum() > 100:                                      # el fondo puede tener un degradado de luz:
        yy, xx = np.nonzero(sel)                             # se modela como un plano suave
        A = np.stack([np.ones(len(xx), np.float32), xx / W, yy / H], 1).astype(np.float32)
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        model = np.empty_like(lab)
        for ch in range(3):
            k = np.linalg.lstsq(A, lab[..., ch][sel], rcond=None)[0].astype(np.float32)
            model[..., ch] = k[0] + k[1] * gx / W + k[2] * gy / H
        d = delta_e(lab, model)
    if np.percentile(d[strip], 60) > 4:                      # la mayoría del borde debe ser de un solo color
        return None
    near = (d < 8).astype(np.uint8)
    _, lbl = cv2.connectedComponents(near, connectivity=8)
    ids = np.unique(lbl[strip & (near > 0)])
    bg = np.isin(lbl, ids[ids > 0])
    # Un fondo de verdad es MUY liso. Una tela arrugada que llena la foto no lo es:
    # así nunca se toma la tela por fondo (ni las manchas por prenda).
    if np.median(d[bg]) > 1.5 or np.percentile(d[bg], 90) > 4:
        return None
    g = (~bg).astype(np.uint8)
    n, lg, st, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
    if n < 2:
        return None
    g = (lg == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)   # la prenda = zona mayor
    cnt, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(g, cnt, -1, 1, -1)                      # rellena huecos (botones, rayas)
    if not 0.10 <= g.mean() <= 0.85:                         # casi todo o casi nada: no es una prenda aislada
        return None
    k = max(4, int(0.022 * min(H, W)))                       # quita el contorno (antialias, sombra, costura)
    g = cv2.erode(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return g > 0
