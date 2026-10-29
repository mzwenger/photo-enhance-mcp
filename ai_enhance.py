"""Modelos de mejora con PyTorch/OpenVINO para Image Kit.

- Retoque automático con 3D LUT adaptativas (Zeng et al., MIT-Adobe FiveK, Apache-2.0).
- Restauración por instrucciones con InstructIR (Conde et al., MIT).
- Coloreado de fotos en blanco y negro con DDColor (Kang et al., Apache-2.0, export ONNX).
- Puntaje de calidad/estética con pyiqa (TOPIQ, MUSIQ, NIMA).

Todo se carga recién la primera vez que se usa, para no demorar el arranque del MCP.
"""

import contextlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np

KIT = Path(__file__).resolve().parent
MODELS = KIT / "models"

_cache: dict = {}


def _quiet():
    """Las librerías imprimen al cargar modelos; en un MCP por stdio eso rompe el protocolo."""
    return contextlib.redirect_stdout(sys.stderr)


def _torch():
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    return torch


# ---------- retoque con 3D LUT ----------

def _lut_models():
    if "lut" not in _cache:
        torch = _torch()
        nn = torch.nn

        def block(i, o, norm):
            layers = [nn.Conv2d(i, o, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
            return layers + ([nn.InstanceNorm2d(o, affine=True)] if norm else [])

        # misma estructura que Classifier en el repo original, para cargar sus pesos
        clf = nn.Sequential(
            nn.Upsample(size=(256, 256), mode="bilinear"),
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.LeakyReLU(0.2), nn.InstanceNorm2d(16, affine=True),
            *block(16, 32, True), *block(32, 64, True), *block(64, 128, True), *block(128, 128, False),
            nn.Dropout(p=0.5), nn.Conv2d(128, 3, 8, padding=0),
        )
        sd = torch.load(MODELS / "lut" / "classifier.pth", map_location="cpu", weights_only=True)
        clf.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()})
        clf.eval()
        luts = torch.load(MODELS / "lut" / "LUTs.pth", map_location="cpu", weights_only=True)
        basis = torch.stack([luts[str(i)]["LUT"] for i in range(3)])  # (3, 3, 33, 33, 33) [lut, canal, b, g, r]
        _cache["lut"] = (clf, basis)
    return _cache["lut"]


def lut_retouch(img_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Aplica el retoque aprendido de fotógrafos profesionales (FiveK, experto C)."""
    torch = _torch()
    clf, basis = _lut_models()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    with torch.no_grad():
        small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        w = clf(torch.from_numpy(small).permute(2, 0, 1)[None]).flatten()
        lut = (w[:, None, None, None, None] * basis).sum(0)[None]  # (1, 3, 33, 33, 33)
        H, W = rgb.shape[:2]
        out = np.empty_like(rgb)
        step = max(1, 4_000_000 // W)  # por franjas para no usar mucha memoria
        for y in range(0, H, step):
            px = torch.from_numpy(rgb[y:y + step]).reshape(1, 1, -1, 1, 3) * 2 - 1  # (x=r, y=g, z=b)
            res = torch.nn.functional.grid_sample(lut, px, mode="bilinear", align_corners=True)
            out[y:y + step] = res.reshape(3, -1).T.reshape(-1, W, 3).numpy()
    out = np.clip(rgb + (out - rgb) * float(np.clip(strength, 0, 1.5)), 0, 1)
    return cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


# ---------- InstructIR ----------

def _instructir():
    if "iir" not in _cache:
        torch = _torch()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoModel, AutoTokenizer
        from ai.instructir.instructir import create_model

        model = create_model(input_channels=3, width=32, enc_blks=[2, 2, 4, 8], middle_blk_num=4,
                             dec_blks=[2, 2, 2, 2], txtdim=256)
        model.load_state_dict(torch.load(MODELS / "im_instructir-7d.pt", map_location="cpu", weights_only=True))
        model.eval()
        head = torch.nn.ModuleDict({"fc1": torch.nn.Linear(384, 256), "fc2": torch.nn.Linear(256, 7)})
        head.load_state_dict(torch.load(MODELS / "lm_instructir-7d.pt", map_location="cpu", weights_only=True))
        with _quiet():
            tok = AutoTokenizer.from_pretrained("TaylorAI/bge-micro-v2")
            lm = AutoModel.from_pretrained("TaylorAI/bge-micro-v2").eval()
        _cache["iir"] = (model, head, tok, lm)
    return _cache["iir"]


IIR_CLASSES = ["ruido", "desenfoque", "lluvia", "neblina", "poca luz", "mejora general", "superresolución"]


def instructir(img_bgr: np.ndarray, instruction: str, tile: int = 768) -> tuple[np.ndarray, str]:
    """Restaura según una instrucción en inglés. Devuelve la imagen y el tipo de
    problema que el modelo entendió."""
    torch = _torch()
    F = torch.nn.functional
    model, head, tok, lm = _instructir()
    with torch.no_grad():
        t = tok([instruction], padding=True, truncation=True, return_tensors="pt")
        tokens = lm(**t)[0]
        m = t["attention_mask"].unsqueeze(-1).float()
        emb = F.normalize((tokens * m).sum(1) / m.sum(1).clamp(min=1e-9), p=2, dim=1)
        txt = F.normalize(head["fc1"](emb), p=2, dim=1)
        understood = IIR_CLASSES[int(head["fc2"](txt).argmax())]

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        H, W = rgb.shape[:2]
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None]
        if H * W <= tile * tile * 1.5:
            out = model(x, txt)[0]
        else:
            # mosaico con solape y mezcla suave para fotos grandes
            ov = 64
            acc = torch.zeros(3, H, W)
            wsum = torch.zeros(1, H, W)
            ramp = torch.ones(tile + 2 * ov)
            ramp[:ov] = torch.linspace(0.05, 1, ov)
            ramp[-ov:] = torch.linspace(1, 0.05, ov)
            for y in range(0, H, tile):
                for x0 in range(0, W, tile):
                    y0, y1 = max(0, y - ov), min(H, y + tile + ov)
                    xa, xb = max(0, x0 - ov), min(W, x0 + tile + ov)
                    o = model(x[:, :, y0:y1, xa:xb], txt)[0]
                    wy = ramp[(ov - (y - y0)):(ov - (y - y0)) + (y1 - y0)]
                    wx = ramp[(ov - (x0 - xa)):(ov - (x0 - xa)) + (xb - xa)]
                    wgt = (wy[:, None] * wx[None, :])[None]
                    acc[:, y0:y1, xa:xb] += o * wgt
                    wsum[:, y0:y1, xa:xb] += wgt
            out = acc / wsum
        out = out.clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR), understood


# ---------- DDColor ----------

def colorize(img_bgr: np.ndarray, compile_model) -> np.ndarray:
    """Colorea una foto en blanco y negro conservando su luminancia original."""
    net = compile_model("ddcolor.onnx")
    img = img_bgr.astype(np.float32) / 255
    H, W = img.shape[:2]
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
    l512 = cv2.cvtColor(cv2.resize(img, (512, 512)), cv2.COLOR_BGR2Lab)[:, :, :1]
    gray = cv2.cvtColor(np.concatenate([l512, np.zeros_like(l512), np.zeros_like(l512)], -1), cv2.COLOR_LAB2RGB)
    ab = next(iter(net([gray.transpose(2, 0, 1)[None]]).values()))[0].transpose(1, 2, 0)
    ab = cv2.resize(ab, (W, H), interpolation=cv2.INTER_CUBIC)
    out = cv2.cvtColor(np.concatenate([orig_l, ab], -1), cv2.COLOR_LAB2BGR)
    return np.clip(out * 255, 0, 255).round().astype(np.uint8)


def looks_grayscale(img_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(cv2.resize(img_bgr, (256, 256)), cv2.COLOR_BGR2HSV)
    return float(np.percentile(hsv[..., 1], 95)) < 30


# ---------- puntaje de calidad ----------

METRICS = {"calidad": "topiq_nr", "estetica": "nima"}


def _metric(name: str):
    key = f"iqa:{name}"
    if key not in _cache:
        _torch()
        import pyiqa
        with _quiet():
            _cache[key] = pyiqa.create_metric(name, device="cpu")
    return _cache[key]


def score(img_bgr: np.ndarray, max_side: int = 768) -> dict:
    """Calidad técnica (TOPIQ, 0-1) y estética (NIMA, 1-10) sobre una versión reducida.
    'total' combina ambas en 0-100."""
    torch = _torch()
    h, w = img_bgr.shape[:2]
    k = max_side / max(h, w)
    if k < 1:
        img_bgr = cv2.resize(img_bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        res = {label: float(_metric(m)(x)) for label, m in METRICS.items()}
    res["total"] = round(50 * res["calidad"] + 50 * (res["estetica"] - 1) / 9, 1)
    res["calidad"] = round(res["calidad"], 3)
    res["estetica"] = round(res["estetica"], 2)
    return res
