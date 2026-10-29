"""Recorte tolerable vs visible.

Lo que se nota de un recorte no es tanto el porcentaje total como el tamaño de la mancha
continua más grande: mil reflejos de un píxel pasan desapercibidos, una zona quemada del
0.3% en el cielo no. Se mide solo el recorte nuevo (el que no estaba en la original),
agrupando píxeles vecinos.
"""

import cv2
import numpy as np

# por tipo: (mancha continua más grande, total) en % de la foto, a partir del cual se nota.
# Reglas prácticas, no una ley: ajustables a gusto.
TOLERANCE = {"luces": (0.1, 1.0), "sombras": (0.3, 2.0), "color": (0.1, 0.5)}
# recorte nuevo por debajo de esto se considera "sin recorte" (ruido de medición)
NONE = {"luces": 0.1, "sombras": 0.1, "color": 0.2}


def masks(img: np.ndarray) -> dict:
    """Píxeles recortados por tipo (por canal, como el histograma del visor)."""
    hi, lo = img >= 253, img <= 2
    h, l = hi.any(2), lo.any(2)
    return {"luces": h, "sombras": l, "color": (h & ~hi.all(2)) | (l & ~lo.all(2))}


def _largest(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return float(stats[1:, cv2.CC_STAT_AREA].max() / mask.size * 100) if n > 1 else 0.0


def assess(img: np.ndarray, orig_masks: dict) -> dict:
    """Recorte nuevo respecto de la original, por tipo: total %, mancha más grande % y nivel
    ("none" sin recorte nuevo, "tol" tolerable, "vis" visible). "level" resume los tres."""
    out, worst = {}, "none"
    rank = {"none": 0, "tol": 1, "vis": 2}
    for k, m in masks(img).items():
        extra = m & ~orig_masks[k]
        total = float(extra.mean() * 100)
        if total <= NONE[k]:
            lvl, big = "none", 0.0
        else:
            big = _largest(extra)
            lb, lt = TOLERANCE[k]
            lvl = "vis" if (big > lb or total > lt) else "tol"
        out[k] = {"total": round(total, 3), "largest": round(big, 3), "level": lvl}
        worst = lvl if rank[lvl] > rank[worst] else worst
    out["level"] = worst
    return out
