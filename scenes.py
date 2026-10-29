"""Mejora natural por escena para Image Kit.

Cada perfil (nublado, noche, atardecer, …) combina los mismos bloques con
distinta intensidad. Los bloques trabajan sobre la luminancia y separan
iluminación (capa base, filtro guiado) de detalle, para cambiar la luz sin
halos ni grano. Las cantidades se expresan relativas al tamaño de la foto, así
una vista previa reducida se parece al resultado a resolución completa.

detect_scene() reconoce la escena con CLIP (zero-shot) más datos EXIF y del
histograma.
"""

import contextlib
import sys
import threading

import cv2
import numpy as np

# ---------- perfiles ----------
# wb: corrección del tinte (0 = respetar la luz de la escena, 1 = neutralizar)
# warmth: + cálido / - frío (unidades del canal b de Lab)
# exposure: cuánto acercar el tono medio a target; shadows/highlights: luz local
# clarity: contraste de detalle medio (negativo suaviza); contrast: curva en S
# dehaze: quitar neblina; vibrance: color en tonos apagados (protege piel)
# denoise: multiplicador sobre el ruido medido; sharpen: enfoque de bordes
PROFILES = {
    "general":   dict(label="General", wb=.5, warmth=0, exposure=.5, target=.46, shadows=.25, highlights=.3,
                      clarity=.15, contrast=.1, dehaze=0, vibrance=.2, denoise=1.0, sharpen=.4),
    "nublado":   dict(label="Día nublado", wb=.6, warmth=4, exposure=.5, target=.48, shadows=.2, highlights=.45,
                      clarity=.3, contrast=.25, dehaze=.15, vibrance=.4, denoise=1.0, sharpen=.5),
    "soleado":   dict(label="Sol fuerte", wb=.3, warmth=0, exposure=.3, target=.46, shadows=.5, highlights=.6,
                      clarity=.1, contrast=0, dehaze=0, vibrance=.15, denoise=.8, sharpen=.4),
    "atardecer": dict(label="Atardecer / hora dorada", wb=0, warmth=3, exposure=.25, target=.42, shadows=.3,
                      highlights=.5, clarity=.1, contrast=.15, dehaze=0, vibrance=.35, denoise=1.0, sharpen=.3),
    "noche":     dict(label="Noche / poca luz", wb=.3, warmth=0, exposure=.4, target=.30, shadows=.25, highlights=.5,
                      clarity=0, contrast=.15, dehaze=0, vibrance=.15, denoise=1.6, sharpen=.2),
    "interior":  dict(label="Interior con luz artificial", wb=.9, warmth=0, exposure=.6, target=.46, shadows=.35,
                      highlights=.3, clarity=.1, contrast=.1, dehaze=0, vibrance=.2, denoise=1.5, sharpen=.3),
    "contraluz": dict(label="Contraluz", wb=.4, warmth=0, exposure=.3, target=.46, shadows=.8, highlights=.6,
                      clarity=.15, contrast=0, dehaze=0, vibrance=.2, denoise=1.2, sharpen=.3),
    "niebla":    dict(label="Niebla / bruma", wb=.4, warmth=0, exposure=.3, target=.46, shadows=.25, highlights=.3,
                      clarity=.2, contrast=.1, dehaze=.4, vibrance=.25, denoise=1.0, sharpen=.4),
    "retrato":   dict(label="Retrato", wb=.5, warmth=1, exposure=.4, target=.48, shadows=.3, highlights=.3,
                      clarity=-.1, contrast=.05, dehaze=0, vibrance=.15, denoise=1.0, sharpen=.25),
}


# ---------- bloques ----------

def _guided(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Filtro guiado (He et al.) con guía en escala de grises: suaviza sin cruzar bordes.
    Con radios grandes usa la versión rápida (He y Sun 2015): los coeficientes se calculan
    a resolución reducida y se aplican sobre la guía completa. Mismo resultado visual con
    mucha menos memoria en fotos grandes, y la misma escala en vista previa y guardado."""
    s = max(1, r // 4)
    if s > 1:
        h, w = I.shape[:2]
        size = (max(1, round(w / s)), max(1, round(h / s)))
        Is = cv2.resize(I, size, interpolation=cv2.INTER_AREA)
        ps = cv2.resize(p, size, interpolation=cv2.INTER_AREA)
        a, b = _guided_coef(Is, ps, max(1, round(r / s)), eps)
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR) * I + cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    a, b = _guided_coef(I, p, r, eps)
    return a * I + b


def _guided_coef(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    del mI, mp
    return box(a), box(b)


def analyze(img: np.ndarray) -> dict:
    # el ruido se mide a la resolución de trabajo (un recorte central), el resto en una versión reducida
    H, W = img.shape[:2]
    c = 1024
    crop = img[max(0, (H - c) // 2):(H + c) // 2, max(0, (W - c) // 2):(W + c) // 2]
    Lc = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255
    g = Lc - cv2.GaussianBlur(Lc, (0, 0), 1.0)
    grad = cv2.magnitude(cv2.Sobel(Lc, cv2.CV_32F, 1, 0), cv2.Sobel(Lc, cv2.CV_32F, 0, 1))
    flat = grad < np.percentile(grad, 30)
    noise = float(1.4826 * np.median(np.abs(g[flat] - np.median(g[flat])))) if flat.any() else 0.0
    k = min(1.0, 1500 / max(H, W))
    small = cv2.resize(img, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255
    return dict(mean=float(L.mean()), median=float(np.median(L)), p01=float(np.percentile(L, 0.5)),
                p99=float(np.percentile(L, 99.7)), clip_hi=float((L > .98).mean()), clip_lo=float((L < .02).mean()),
                noise=noise, dark_frac=float((L < .25).mean()), bright_frac=float((L > .8).mean()))


def _soft(x: np.ndarray, lo: float = .04, hi: float = .96) -> np.ndarray:
    """Comprime suavemente los extremos en lugar de recortarlos (pie y hombro, como una
    curva de película): entre lo y hi no cambia nada; por fuera los tonos se acercan a 0 o
    a 1 sin llegar nunca, así no se pierden gradaciones en sombras ni en luces."""
    y = np.where(x < lo, lo * np.exp((np.minimum(x, lo) - lo) / lo), x)
    return np.where(y > hi, 1 - (1 - hi) * np.exp(-(np.maximum(y, hi) - hi) / (1 - hi)), y).astype(np.float32)


def _neutral_shift(lab: np.ndarray) -> tuple[float, float]:
    """Tinte estimado con píxeles casi grises (no con el promedio de la foto), para
    que un pasto verde o un cielo azul no se tomen como error de color."""
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    chroma = np.hypot(a, b)
    mask = (L > 40) & (L < 235) & (chroma < np.percentile(chroma, 20)) & (chroma < 18)
    if mask.sum() < 0.01 * L.size:
        return 0.0, 0.0
    return float(np.median(a[mask])), float(np.median(b[mask]))


def _skin_mask(lab: np.ndarray) -> np.ndarray:
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    hue = np.degrees(np.arctan2(b, a))
    chroma = np.hypot(a, b)
    m = ((hue > 25) & (hue < 75) & (chroma > 8) & (chroma < 45) & (lab[..., 0] > 50)).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 3)


def _dehaze(rgb: np.ndarray, amount: float, r: int) -> np.ndarray:
    """Prior del canal oscuro (He et al.) con transmisión refinada por filtro guiado.
    Luz ambiente y transmisión se estiman a ~1000 px (ventana de 15 px a esa escala)."""
    H, W = rgb.shape[:2]
    k = min(1.0, 1000 / max(H, W))
    small = cv2.resize(rgb, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA) if k < 1 else rgb
    dark = cv2.erode(small.min(axis=2), np.ones((15, 15), np.uint8))
    flat = dark.ravel()
    n = max(1, flat.size // 1000)
    idx = np.argpartition(flat, flat.size - n)[-n:]
    A = small.reshape(-1, 3)[idx].mean(0).clip(.5, 1).astype(np.float32)
    t = (1 - .95 * cv2.erode((small / A).min(axis=2), np.ones((15, 15), np.uint8))).astype(np.float32)
    if k < 1:
        t = cv2.resize(t, (W, H), interpolation=cv2.INTER_LINEAR)
    t = _guided(rgb.mean(2), t, r, 1e-3).clip(.35, 1)[..., None]
    return rgb + (((rgb - A) / t + A).clip(0, 1) - rgb) * amount


def enhance(img: np.ndarray, scene: str = "general", amount: float = 1.0, scale: float = 1.0) -> np.ndarray:
    """Aplica el perfil de escena. scale = ancho de img / ancho de la foto completa."""
    P = PROFILES.get(scene, PROFILES["general"])
    k = float(np.clip(amount, 0, 1.5))
    if k == 0:
        return img
    H, W = img.shape[:2]
    full_w = W / max(scale, 1e-6)
    r = max(2, round(min(H, W) * .015))  # radio de la capa base, relativo al tamaño
    st = analyze(img)

    # todo en flotante (Lab de OpenCV en float: L 0-100, a/b centrados en 0): pasar por
    # 8 bits antes de estirar los tonos deja niveles vacíos (histograma en "peine", bandas)
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    L = lab[..., 0] / 100
    a, b = lab[..., 1], lab[..., 2]
    del lab

    # 1) ruido: luminancia con filtro guiado según el ruido medido; color más fuerte
    dn = P["denoise"] * k
    if dn > 0 and st["noise"] > .004:
        eps = (st["noise"] * 2.2 * dn) ** 2
        L = _guided(L, L, 1 if dn < 1.5 else 2, eps)
        rc = max(2, round(r * .5 * dn))
        # eps en unidades de la guía (L, 0-1): bajo para que el color no cruce bordes
        a = _guided(L, a, rc, 2e-3 * dn)
        b = _guided(L, b, rc, 2e-3 * dn)

    # 2) color de la luz
    if P["wb"] > 0:
        da, db = _neutral_shift(np.dstack([L * 255, a + 128, b + 128]))
        lim = 10
        a -= np.clip(da, -lim, lim) * P["wb"] * min(k, 1)
        b -= np.clip(db, -lim, lim) * P["wb"] * min(k, 1)
    b += P["warmth"] * k

    # 3) neblina (en RGB, antes del tono)
    if P["dehaze"] > 0:
        rgb = cv2.cvtColor(np.dstack([L * 100, a, b]).astype(np.float32), cv2.COLOR_LAB2RGB).clip(0, 1)
        med0 = float(np.median(L))
        rgb = _dehaze(rgb, P["dehaze"] * k, r)
        lab2 = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
        L, a, b = lab2[..., 0] / 100, lab2[..., 1], lab2[..., 2]
        # quitar neblina oscurece: se devuelve el 70% del brillo medio perdido
        med1 = float(np.median(L))
        if .02 < med1 < med0 < .98:
            L = L ** (1 + (np.log(med0) / np.log(med1) - 1) * .7)

    # 4) puntos negro/blanco sin recortar más de lo que ya estaba
    lo, hi = np.percentile(L, .3), np.percentile(L, 99.8)
    lo = min(lo, .04) * min(k, 1)
    hi = 1 - (1 - max(hi, .9)) * min(k, 1)
    L = _soft((L - lo) / max(hi - lo, .2))

    # 5) base/detalle: la luz se ajusta en la base, el detalle se conserva
    base = _guided(L, L, r, 1e-2)
    detail = L - base
    med = float(np.median(base))
    if P["exposure"] > 0 and 0.02 < med < .98:
        g = np.log(P["target"]) / np.log(med)  # gamma que lleva la mediana al objetivo
        if g > 1:
            # foto más clara que el objetivo: se oscurece poco (una escena luminosa debe
            # seguir siéndolo) y solo en medios y luces, para no hundir las sombras
            g = float(min(1 + (g - 1) * P["exposure"] * k * .4, 1.2))
            w = np.clip(base / .35, 0, 1) ** 2
            base = base + (base ** g - base) * w
        else:
            g = float(max(1 + (g - 1) * P["exposure"] * k, .6))
            base = base ** g
    lo0 = float(np.percentile(base, .5))
    sh, hl = P["shadows"] * k, P["highlights"] * k
    base = base + sh * .9 * base * (1 - base) ** 2 - hl * .9 * base ** 2 * (1 - base)
    if P["contrast"]:
        s = base * base * (3 - 2 * base)  # curva S suave
        base = base + (s - base) * P["contrast"] * k
    # los negros vuelven a su nivel: aclarar sombras no debe dejar la foto lavada
    lift = float(np.percentile(base, .5)) - lo0
    if lift > 0:
        base = _soft((base - lift) / (1 - lift))
    # el detalle no crece en zonas planas (ruido) y crece en zonas con textura
    gate = cv2.GaussianBlur(np.abs(detail), (0, 0), 2)
    gate = np.clip(gate / (3 * max(st["noise"], .003)) - .5, 0, 1)
    L = _soft(base + detail * (1 + P["clarity"] * k * (gate if P["clarity"] > 0 else 1)))
    del base, detail, gate

    # 6) vibración: más en colores apagados, menos en piel y en lo ya saturado
    if P["vibrance"] > 0:
        chroma = np.hypot(a, b)
        skin = _skin_mask(np.dstack([L * 255, a + 128, b + 128]))
        boost = 1 + P["vibrance"] * k * (1 - (chroma / 60).clip(0, 1)) * (1 - .7 * skin)
        a, b = a * boost, b * boost

    # 7) enfoque solo en bordes reales, radio según el tamaño de la foto
    if P["sharpen"] > 0:
        sigma = max((1.0 + full_w / 3000) * scale, .6)
        hp = L - cv2.GaussianBlur(L, (0, 0), sigma)
        edge = cv2.GaussianBlur(np.abs(hp), (0, 0), sigma)
        mask = np.clip(edge / (2.5 * max(st["noise"], .003)) - 1, 0, 1)
        L = _soft(L + 1.2 * P["sharpen"] * k * hp * mask)

    out = cv2.cvtColor(np.dstack([L * 100, a, b]).astype(np.float32), cv2.COLOR_LAB2BGR)
    # tramado de medio nivel al volver a 8 bits: evita bandas en cielos y degradados
    noise = np.random.default_rng(0).uniform(-.5, .5, out.shape[:2])[..., None]
    return (out * 255 + noise).round().clip(0, 255).astype(np.uint8)


# ---------- detección de escena ----------

PROMPTS = {
    "noche": ["a photo taken at night", "a night scene with artificial lights", "a dark photo taken in low light"],
    "nublado": ["a photo taken on a cloudy overcast day", "an outdoor scene under a grey sky with soft light"],
    "soleado": ["a photo taken on a bright sunny day with hard shadows", "a sunny outdoor scene with blue sky"],
    "atardecer": ["a photo of a sunset", "a photo taken during golden hour with warm light"],
    "interior": ["a photo taken indoors under warm artificial light", "a photo of a room inside a house"],
    "contraluz": ["a backlit photo with the subject in shadow against a bright background",
                  "a silhouette against a bright sky"],
    "niebla": ["a foggy landscape", "a hazy misty photo with low contrast"],
    "retrato": ["a portrait photo of a person", "a close-up photo of a person's face"],
}
_clip = {}
_clip_lock = threading.Lock()  # el lote procesa varias fotos a la vez


def _clip_model():
    with _clip_lock:
        return _clip_load()


def _clip_load():
    if "m" not in _clip:
        import clip
        import torch
        with contextlib.redirect_stdout(sys.stderr):
            model, pre = clip.load("ViT-B/32", device="cpu")
        names, texts = [], []
        for scene, ps in PROMPTS.items():
            names += [scene] * len(ps)
            texts += ps
        with torch.no_grad():
            t = model.encode_text(clip.tokenize(texts)).float()
            t /= t.norm(dim=-1, keepdim=True)
        _clip.update(m=model, pre=pre, t=t, names=names)
    return _clip


def unload() -> None:
    with _clip_lock:
        _clip.clear()


def detect_scene(img: np.ndarray, exif: dict | None = None) -> tuple[str, dict]:
    """Devuelve la escena más probable y las probabilidades de cada una."""
    import torch
    from PIL import Image as PILImage
    c = _clip_model()
    small = cv2.resize(img, (336, round(336 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    x = c["pre"](PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))[None]
    with torch.no_grad():
        f = c["m"].encode_image(x).float()
        f /= f.norm(dim=-1, keepdim=True)
        logits = (100 * f @ c["t"].T)[0].numpy()
    scenes = list(PROMPTS)
    per = np.array([max(logits[i] for i, n in enumerate(c["names"]) if n == s) for s in scenes])
    prob = np.exp(per - per.max())
    prob /= prob.sum()
    pr = dict(zip(scenes, prob))

    # datos de la foto que CLIP no ve bien
    st = analyze(img)
    ex = exif or {}
    t, iso = ex.get("exposure_time"), ex.get("iso")
    if st["mean"] > .45 and (t is None or t < 1 / 60):
        pr["noche"] *= .2
    if (t and t >= 1 / 15) or (iso and iso >= 3200 and st["mean"] < .35):
        pr["noche"] *= 2.5
    if not (st["dark_frac"] > .25 and st["bright_frac"] > .15):
        pr["contraluz"] *= .4
    s = sum(pr.values())
    pr = {k: round(float(v / s), 3) for k, v in sorted(pr.items(), key=lambda kv: -kv[1])}
    best = next(iter(pr))
    return (best if pr[best] >= .3 else "general"), pr


def read_exif(path) -> dict:
    from PIL import Image as PILImage
    try:
        with PILImage.open(path) as im:
            sub = im.getexif().get_ifd(0x8769)
        t, iso = sub.get(0x829A), sub.get(0x8827)
        return {"exposure_time": float(t) if t else None, "iso": int(iso) if iso else None}
    except Exception:
        return {}
