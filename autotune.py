"""Búsqueda determinista de los parámetros óptimos de un perfil de escena.

Reproduce el procedimiento manual (analizar, probar variantes, medir, descartar las que se
pasan, revisar al 100%) con reglas fijas, para que la misma foto dé siempre el mismo resultado:

1. Parte del perfil de la escena detectada.
2. Búsqueda por coordenadas, en orden fijo, sobre una versión de 1200 px.
3. Cada candidato recibe un puntaje: calidad/estética (pyiqa) + detalle ganado por zona
   - penalizaciones por recortes nuevos, tinte en los blancos, sobresaturación, cambios de
   brillo por zona y grano en zonas lisas (medido al 100% sobre un recorte de la foto).
4. Ruido y nitidez se eligen al final mirando ese recorte al 100%, donde se ve el grano.
"""

import cv2
import numpy as np

import ai_enhance as A
import scenes as SC

TUNE_SIDE = 900
CROP = 640
N_CROPS = 3
IQA_SIDE = 512
# cada parámetro que se aleja del perfil "cuesta": solo se mueve si la mejora es clara (natural)
REG = 5.0
# orden y valores fijos: parte del determinismo
SEARCH = [("highlights", [.3, .45, .6, .75]), ("shadows", [.15, .3, .45, .6]), ("exposure", [.3, .45, .6]),
          ("clarity", [.15, .3, .45]), ("contrast", [.1, .25, .4]), ("vibrance", [.2, .3, .4, .55]),
          ("dehaze", [0, .1, .25]), ("warmth", [0, 1, 2.5])]
FINISH = [("denoise", [1.0, 1.4, 1.8]), ("sharpen", [.4, .6, .8])]


def _small(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    k = TUNE_SIDE / max(h, w)
    return cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img


def _lab(img):
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return lab[..., 0] / 100, lab[..., 2]


def _lab_full(img):
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return lab[..., 0] / 100, lab[..., 1], lab[..., 2]


def color_clip(img: np.ndarray) -> float:
    """% de píxeles con algún canal en el tope pero no todos: se perdió el matiz aunque la
    luminosidad parezca correcta (cielo azul plano, flor roja sin degradé)."""
    hi, lo = img >= 253, img <= 2
    part = (hi.any(2) & ~hi.all(2)) | (lo.any(2) & ~lo.all(2))
    return float(part.mean() * 100)


def _local_detail(L):
    m = cv2.blur(L, (9, 9))
    return np.sqrt(np.maximum(cv2.blur(L * L, (9, 9)) - m * m, 0))


def _grain(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float((g - cv2.GaussianBlur(g, (0, 0), 2)).std())


class Reference:
    """Medidas de la original contra las que se compara cada candidato."""

    def __init__(self, full: np.ndarray):
        self.full_w = full.shape[1]
        self.small = _small(full)
        self.scale = self.small.shape[1] / self.full_w
        L, a, b = _lab_full(self.small)
        self.zones = {"sombras": L < .28, "medios": (L >= .28) & (L < .72), "luces": L >= .72}
        self.whites = L > .85
        chroma = np.hypot(a, b)
        self.colored = chroma > 10  # píxeles con color definido, para medir el giro de tono
        self.hue = np.arctan2(b, a)
        self.stats = self._stats(self.small)
        self.crops = self._flat_crops(full)
        self.crop_grain = [_grain(c) for c in self.crops]

    def _flat_crops(self, full):
        """Zonas lisas y claras (cielo, pared) donde el grano se nota más, a resolución completa."""
        H, W = full.shape[:2]
        if min(H, W) < 2 * CROP:
            return []
        g = cv2.cvtColor(self.small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        k = self.scale
        win = max(8, round(CROP * k))
        act = cv2.blur(grad, (win, win))
        bright = cv2.blur(g, (win, win)) > 100
        act[~bright] = np.inf
        half = win // 2
        act[:half], act[-half:], act[:, :half], act[:, -half:] = np.inf, np.inf, np.inf, np.inf
        crops = []
        for _ in range(N_CROPS):  # las zonas más lisas, sin superponerse
            y, x = np.unravel_index(np.argmin(act), act.shape)
            if not np.isfinite(act[y, x]):
                break
            cy, cx = round(y / k), round(x / k)
            y0, x0 = int(np.clip(cy - CROP // 2, 0, H - CROP)), int(np.clip(cx - CROP // 2, 0, W - CROP))
            crops.append(full[y0:y0 + CROP, x0:x0 + CROP].copy())
            act[max(0, y - win):y + win, max(0, x - win):x + win] = np.inf
        return crops

    def _stats(self, img):
        L, a, b = _lab_full(img)
        det = _local_detail(L)
        cdet = (_local_detail(a) + _local_detail(b)) / 2  # detalle de color (variación de a/b)
        dh = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(b, a) - self.hue)))))
        hue_shift = float(np.median(dh[self.colored])) if self.colored.mean() > .02 else 0.0
        sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32).mean()
        return dict(neg=float((L <= .01).mean() * 100), alt=float((L >= .99).mean() * 100), sat=float(sat),
                    b_whites=float(b[self.whites].mean()) if self.whites.mean() > .02 else None,
                    zone_mean={z: float(L[m].mean()) for z, m in self.zones.items() if m.mean() > .03},
                    zone_det={z: float(det[m].mean()) for z, m in self.zones.items() if m.mean() > .03},
                    color_clip=color_clip(img), color_det=float(cdet.mean()), hue_shift=hue_shift)


def evaluate(ref: Reference, img: np.ndarray) -> tuple[float, dict]:
    """Puntaje de un candidato (más alto = mejor) y su desglose."""
    s, o = ref._stats(img), ref.stats
    q = A.score(img, IQA_SIDE)["total"]
    gains = [np.clip(np.log2(s["zone_det"][z] / max(o["zone_det"][z], 1e-6)), -1, 1) for z in o["zone_det"]]
    detail = 8 * float(np.mean(gains)) if gains else 0.0
    pen = {}
    pen["recortes"] = 15 * max(0, s["neg"] - o["neg"] - .1) + 15 * max(0, s["alt"] - o["alt"] - .1)
    pen["saturación"] = 20 * max(0, s["sat"] / max(o["sat"], 1) - 1.5)
    pen["tinte blancos"] = (2.5 * max(0, abs(s["b_whites"] - o["b_whites"]) - 1.5)
                            if o["b_whites"] is not None else 0.0)
    # brillo por zona (L 0-100): oscurecer sombras se tolera poco; aclararlas o bajar luces, más
    tol = {"sombras": (4, 12), "medios": (8, 10), "luces": (10, 6)}
    z_pen = 0.0
    for z, m0 in o["zone_mean"].items():
        d = (s["zone_mean"][z] - m0) * 100
        down, up = tol[z]
        z_pen += .5 * max(0, -d - down) + .5 * max(0, d - up)
    pen["brillo por zona"] = z_pen
    # pérdida de información de color
    pen["color recortado"] = 15 * max(0, s["color_clip"] - o["color_clip"] - .2)
    pen["detalle de color"] = 30 * max(0, .85 - s["color_det"] / max(o["color_det"], 1e-6))
    pen["giro de tono"] = 1.0 * max(0, s["hue_shift"] - 4)
    total = q + detail - sum(pen.values())
    return total, dict(calidad=q, detalle=detail, **pen)


def _grain_penalty(ref: Reference, scene: str, params: dict, amount: float = 1.0) -> tuple[float, float]:
    if not ref.crops:
        return 0.0, 1.0
    ratios = [_grain(SC.enhance(c, scene, amount, 1.0, params, full_width=ref.full_w)) / max(g0, 1e-6)
              for c, g0 in zip(ref.crops, ref.crop_grain)]
    ratio = float(np.mean(ratios))
    return 40 * max(0, ratio - 1.2), ratio


def tune(refs: list[Reference], scene: str, progress=None) -> dict:
    """Busca los parámetros que maximizan el puntaje medio sobre refs (una foto o un grupo)."""
    SC.sync_profiles()
    base = SC.PROFILES.get(scene, SC.PROFILES["general"])
    cur = {k: float(base[k]) for k in SC.RANGES}
    cache = {}

    def score(params, amount=1.0):
        key = tuple(round(params[k], 4) for k in SC.RANGES) + (amount,)
        if key not in cache:
            vals = [evaluate(r, SC.enhance(r.small, scene, amount, r.scale, params)) for r in refs]
            reg = REG * sum(abs(params[k] - float(base[k])) / (SC.RANGES[k][1] - SC.RANGES[k][0])
                            for k in SC.RANGES)
            # el grano al 100% lo suben claridad, bruma y nitidez: se mide en cada candidato
            gp = [_grain_penalty(r, scene, params, amount) for r in refs]
            grain_pen = float(np.mean([g[0] for g in gp]))
            total = float(np.mean([v[0] for v in vals])) - reg - grain_pen
            parts = {k: float(np.mean([v[1][k] for v in vals])) for k in vals[0][1]}
            parts["cambio vs perfil"] = reg
            parts["grano"] = grain_pen
            parts["grano_ratio"] = float(np.mean([g[1] for g in gp]))
            cache[key] = (total, parts)
            if progress:
                progress(len(cache))
        return cache[key]

    best, _ = score(cur)
    for _ in range(2):  # dos pasadas por coordenadas, en orden fijo
        changed = False
        for name, values in SEARCH:
            for v in sorted(set(values + [round(float(base[name]), 4)])):
                cand = {**cur, name: v}
                t, _ = score(cand)
                if t > best + 1e-9:
                    best, cur, changed = t, cand, True
        if not changed:
            break
    # ruido y nitidez al final, ya con luz y color decididos
    for dn in FINISH[0][1]:
        for sh in FINISH[1][1]:
            cand = {**cur, "denoise": dn, "sharpen": sh}
            t, _ = score(cand)
            if t > best + 1e-9:
                best, cur = t, cand
    # intensidad: si el ajuste completo no convence, probar más suave o dejar la foto como está
    amount = 1.0
    for amt in (.75, .5):
        t, _ = score(cur, amt)
        if t > best + 1e-9:
            best, amount = t, amt
    orig = [evaluate(r, r.small) for r in refs]
    orig_total = float(np.mean([o[0] for o in orig]))
    if orig_total >= best - 1.0:  # mejora menor a 1 punto: no vale la pena tocar la foto
        amount = 0.0
    finish_best = best
    total, parts = score(cur, amount) if amount else (orig_total, {
        **{k: float(np.mean([o[1][k] for o in orig])) for k in orig[0][1]},
        "cambio vs perfil": 0.0, "grano": 0.0, "grano_ratio": 1.0})
    grain_ratio = parts["grano_ratio"]
    return dict(scene=scene, params=cur, amount=amount, score=finish_best, parts=parts, grain_ratio=grain_ratio,
                evaluated=len(cache),
                quality_before=float(np.mean([o[1]["calidad"] for o in orig])),
                quality_after=parts["calidad"])


def report(res: dict, base: dict) -> str:
    changed = {k: v for k, v in res["params"].items() if abs(v - float(base[k])) > 1e-6}
    if res["amount"] == 0:
        return (f"Escena: {res['scene']}. Ningún ajuste mejora la foto (calidad/estética "
                f"{res['quality_before']:.1f}): se deja como está. Combinaciones evaluadas: {res['evaluated']}.")
    lines = [f"Escena: {res['scene']}. Calidad/estética {res['quality_before']:.1f} -> {res['quality_after']:.1f} "
             f"(detalle {res['parts']['detalle']:+.1f})"
             + (f", intensidad {res['amount'] * 100:.0f}%." if res["amount"] < 1 else "."),
             "Cambios respecto del perfil: " + (", ".join(f"{k} {float(base[k]):g}->{v:g}" for k, v in changed.items())
                                                 or "ninguno"),
             "Penalizaciones: " + ", ".join(f"{k} {res['parts'][k]:.1f}" for k in
                                            ["recortes", "saturación", "tinte blancos", "brillo por zona",
                                             "color recortado", "detalle de color", "giro de tono",
                                             "grano", "cambio vs perfil"]) +
             f"; grano en zona lisa al 100%: x{res['grain_ratio']:.2f}.",
             f"Combinaciones evaluadas: {res['evaluated']}."]
    return "\n".join(lines)
