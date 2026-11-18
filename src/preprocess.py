"""Carga, reducción de ruido y corrección de iluminación."""
import cv2
import numpy as np


def load_bgr(file_bytes, max_side=640):
    """Decodifica y reduce la imagen (ahorra RAM). Respeta la orientación EXIF (fotos de celular).
    Los PNG con fondo transparente (típicos de catálogo) se pegan sobre blanco."""
    buf = np.frombuffer(file_bytes, np.uint8)
    if file_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is not None and img.dtype == np.uint16:
            img = (img / 257).astype(np.uint8)
        if img is not None and img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img is not None and img.shape[2] == 4:
            a = img[..., 3:4].astype(np.float32) / 255.0
            img = (img[..., :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    else:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo leer la imagen")
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def crop_pct(bgr, left=0, right=0, top=0, bottom=0):
    """Recorta bordes en % (quita manos, mangas, pared)."""
    h, w = bgr.shape[:2]
    return bgr[int(h * top / 100): h - int(h * bottom / 100),
               int(w * left / 100): w - int(w * right / 100)].copy()


def denoise(bgr):
    """Bilateral: quita ruido sin borrar los bordes de las manchas."""
    return cv2.bilateralFilter(bgr, 7, 35, 5)


def local_background(chan, mask, size=128, frac=0.10, iters=4):
    """Color normal LOCAL de una prenda aislada: promedio suave (solo dentro de la prenda)
    que en cada vuelta deja fuera los puntos raros (manchas). Sigue el sombreado de mangas
    y pliegues, que un polinomio global no alcanza a seguir."""
    h, w = chan.shape
    s = size / max(h, w)
    dsz = (max(int(w * s), 8), max(int(h * s), 8))
    c = cv2.resize(chan, dsz, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), dsz, interpolation=cv2.INTER_NEAREST) > 0
    sig = frac * max(dsz)
    wgt = m.astype(np.float32)
    for _ in range(iters):
        num = cv2.GaussianBlur(c * wgt, (0, 0), sig)
        den = cv2.GaussianBlur(wgt, (0, 0), sig)
        bg = num / np.maximum(den, 1e-4)
        r = c - bg
        s_r = 1.4826 * np.median(np.abs(r[wgt > 0] - np.median(r[wgt > 0]))) + 1e-3
        wgt = (m & (np.abs(r) < 2.0 * s_r)).astype(np.float32)
        if wgt.sum() < 0.3 * m.sum():
            wgt = m.astype(np.float32)
            break
    bg = np.where(den > 1e-3, bg, c)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def background(chan, size=64, iters=6, mask=None):
    """Fondo suave del canal con un ajuste de superficie (polinomio grado 2) ROBUSTO.
    En cada vuelta descarta los puntos que no encajan (manchas) y reajusta, así una
    mancha grande no contamina el 'color normal' de la tela. Con `mask`, solo usa la prenda."""
    if mask is not None and mask.sum() >= 50:
        return local_background(chan, mask)
    h, w = chan.shape
    s = size / max(h, w)
    small = cv2.resize(chan, (max(int(w * s), 8), max(int(h * s), 8)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape

    def terms(yy, xx):
        return np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1)

    ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float32)
    A = terms(ys / sh - 0.5, xs / sw - 0.5).reshape(-1, 6)
    z = small.reshape(-1)
    ok = np.ones(len(z), bool)
    if mask is not None:
        ms = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
        if ms.sum() >= 50:
            ok = ms
    inl = ok.copy()
    for _ in range(iters):
        coef = np.linalg.lstsq(A[inl], z[inl], rcond=None)[0]
        r = z - A @ coef
        sig = 1.4826 * np.median(np.abs(r[inl] - np.median(r[inl]))) + 1e-3
        new_inl = (np.abs(r) < 2.0 * sig) & ok
        if new_inl.sum() < 0.3 * ok.sum():
            break
        inl = new_inl
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    full = terms(yy / h - 0.5, xx / w - 0.5) @ coef.astype(np.float32)
    return full.astype(np.float32)


def flatten_illumination(lab, mask=None):
    """Quita degradados de luz/sombras suaves del canal L."""
    L = lab[..., 0]
    bgL = background(L, mask=mask)
    out = lab.copy()
    out[..., 0] = L - bgL + np.median(bgL if mask is None else bgL[mask])
    return out
