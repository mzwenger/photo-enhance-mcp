"""Image Kit: servidor MCP local para editar y mejorar fotos.

Herramientas: info, mejora automática, ajustes, reducción de ruido, reescalado
(Real-ESRGAN/Vulkan), quitar fondo (rembg), borrar objetos (LaMa), restaurar
caras (GFPGAN), comparar original/editada (imagen o galería HTML) y mejora con IA
(retoque LUT, InstructIR, DDColor, puntaje de calidad y elección automática de la
mejor versión; ver ai_enhance.py). La inferencia ONNX corre con OpenVINO en el mejor dispositivo
disponible (GPU/NPU si sus drivers están instalados, si no CPU).
"""

import ctypes
import functools
import gc
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import webbrowser
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from PIL import Image as PILImage, ImageOps

KIT = Path(__file__).resolve().parent
MODELS = KIT / "models"
REALESRGAN = KIT / "bin" / "realesrgan" / "realesrgan-ncnn-vulkan"
os.environ.setdefault("U2NET_HOME", str(MODELS / "rembg"))

mcp = MCPServer("image-kit")

# Los modelos quedan cargados para que las llamadas siguientes sean rápidas, pero el servidor
# vive mucho tiempo: si su memoria supera este límite se descargan (y se recargan al usarlos).
MAX_RAM_GB = float(os.environ.get("IMAGE_KIT_MAX_RAM_GB", "3"))


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            return int(f.read().split("VmRSS:")[1].split()[0]) / 1024 / 1024
    except (OSError, IndexError, ValueError):
        return 0.0


def _trim_memory() -> None:
    if _rss_gb() <= MAX_RAM_GB:
        return
    with _load_lock:
        _compiled.clear()
        _rembg_sessions.clear()
    for mod in ("ai_enhance", "scenes"):
        if mod in sys.modules:
            sys.modules[mod].unload()
    gc.collect()
    try:  # devolver al sistema la memoria liberada
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _tool(f):
    """Registra la herramienta en el MCP y libera modelos al terminar si hace falta."""
    @functools.wraps(f)
    def run(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        finally:
            _trim_memory()
    return mcp.tool()(run)


_ov_core = None
_compiled = {}
_rembg_sessions = {}
_load_lock = threading.Lock()  # el lote procesa varias fotos a la vez


# ---------- utilidades ----------

def _device() -> str:
    global _ov_core
    import openvino as ov
    if _ov_core is None:
        _ov_core = ov.Core()
    devs = _ov_core.available_devices
    return os.environ.get("IMAGE_KIT_DEVICE") or ("GPU" if "GPU" in devs else "CPU")


def _model(name: str):
    with _load_lock:
        if name not in _compiled:
            _device()
            _compiled[name] = _ov_core.compile_model(str(MODELS / name), _device())
        return _compiled[name]


def _load(path: str) -> np.ndarray:
    """Carga como BGR uint8 respetando la orientación EXIF."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"No existe la imagen: {p}")
    im = ImageOps.exif_transpose(PILImage.open(p)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _out_path(src: str, suffix: str, output_path: str | None, ext: str | None = None) -> Path:
    if output_path:
        out = Path(output_path).expanduser()
    else:
        s = Path(src).expanduser()
        out = s.with_name(f"{s.stem}_{suffix}{ext or s.suffix or '.png'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _save(img: np.ndarray, out: Path) -> None:
    if img.ndim == 3 and img.shape[2] == 4:
        PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)).save(out)
    else:
        rgb = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        kw = {"quality": 95} if out.suffix.lower() in (".jpg", ".jpeg", ".webp") else {}
        rgb.save(out, **kw)


def _preview(img: np.ndarray, max_side: int = 768) -> Image:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    im = PILImage.fromarray(rgb)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    if im.mode == "RGBA":
        im.save(buf, "PNG")
        return Image(data=buf.getvalue(), format="png")
    im.save(buf, "JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg")


def _result(img: np.ndarray, out: Path, note: str = ""):
    h, w = img.shape[:2]
    msg = f"Guardado en {out} ({w}x{h})."
    return [msg + (f" {note}" if note else ""), _preview(img)]


# ---------- herramientas ----------

@_tool
def image_info(path: str):
    """Devuelve tamaño, formato y una vista previa de la imagen."""
    p = Path(path).expanduser()
    with PILImage.open(p) as im:
        fmt, mode = im.format, im.mode
    img = _load(path)
    h, w = img.shape[:2]
    return [f"{p}: {w}x{h}, formato {fmt}, modo {mode}, {p.stat().st_size // 1024} KB. "
            f"Dispositivo de inferencia: {_device()}.", _preview(img)]


def _enhance(img: np.ndarray, strength: float, scale: float = 1.0) -> np.ndarray:
    """scale: ancho de img / ancho de la foto completa, para que la nitidez de una
    vista reducida se parezca a la del resultado final."""
    img = img.astype(np.float32)
    s = float(np.clip(strength, 0, 2))
    # balance de blancos (mundo gris), atenuado
    means = img.reshape(-1, 3).mean(0)
    gain = means.mean() / np.maximum(means, 1e-3)
    img = np.clip(img * (1 + (gain - 1) * 0.6 * min(s, 1)), 0, 255).astype(np.uint8)
    # contraste local en luminancia
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=1.0 + 1.5 * s, tileGridSize=(8, 8)).apply(l)
    l = cv2.addWeighted(l, 1 - min(s, 1) * 0.7, l2, min(s, 1) * 0.7, 0)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    # saturación
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + 0.15 * s), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # nitidez: mismo enfoque que el control manual, visible igual en vista previa y guardado
    return _sharpen(img, 0.2 * s, scale)


def _adjust(img: np.ndarray, brightness: float = 0, contrast: float = 0, saturation: float = 0,
            warmth: float = 0, gamma: float = 1.0, sharpness: float = 0, scale: float = 1.0) -> np.ndarray:
    img = img.astype(np.float32)
    if gamma != 1.0:
        img = 255 * (img / 255) ** (1 / max(gamma, 0.05))
    img = (img - 127.5) * (1 + contrast / 100) + 127.5 + brightness * 1.275
    if warmth:
        img[..., 2] += warmth * 0.4
        img[..., 0] -= warmth * 0.4
    img = np.clip(img, 0, 255).astype(np.uint8)
    if saturation:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + saturation / 100), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if sharpness > 0:
        img = _sharpen(img, sharpness / 100, scale)
    return img


def _sharpen(img: np.ndarray, amount: float, scale: float = 1.0) -> np.ndarray:
    """Máscara de enfoque sobre la luminancia (sin halos de color). El radio crece
    con el tamaño de la foto (3 px a 6000 px de ancho) para que se note igual en
    fotos grandes, y en vistas reducidas no baja de 0.6 px para que el efecto se vea."""
    full_w = img.shape[1] / max(scale, 1e-6)
    sigma = max((1.0 + full_w / 3000) * scale, 0.6)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[..., 0].astype(np.float32)
    k = 1.5 * amount
    l = l + k * (l - cv2.GaussianBlur(l, (0, 0), sigma))
    lab[..., 0] = np.clip(l, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@_tool
def auto_enhance(path: str, strength: float = 1.0, output_path: str | None = None):
    """Mejora automática: balance de blancos, contraste local, color y nitidez.

    strength: 0.0 a 2.0 (1.0 = moderado).
    """
    img = _enhance(_load(path), float(np.clip(strength, 0, 2)))
    out = _out_path(path, "enhanced", output_path)
    _save(img, out)
    return _result(img, out)


@_tool
def adjust(path: str, brightness: float = 0, contrast: float = 0, saturation: float = 0,
           warmth: float = 0, gamma: float = 1.0, sharpness: float = 0,
           output_path: str | None = None):
    """Ajustes manuales. brightness/contrast/saturation/warmth en -100..100,
    gamma 0.3..3 (<1 aclara sombras), sharpness 0..100."""
    p = _params(dict(brightness=brightness, contrast=contrast, saturation=saturation, warmth=warmth,
                     gamma=gamma, sharpness=sharpness))
    img = _adjust(_load(path), p["brightness"], p["contrast"], p["saturation"], p["warmth"], p["gamma"],
                  p["sharpness"])
    out = _out_path(path, "adjusted", output_path)
    _save(img, out)
    return _result(img, out)


@_tool
def denoise(path: str, strength: int = 7, output_path: str | None = None):
    """Reduce el ruido digital conservando bordes. strength 3..20 (7 = moderado)."""
    img = _load(path)
    s = int(np.clip(strength, 3, 20))
    img = cv2.fastNlMeansDenoisingColored(img, None, s, s, 7, 21)
    out = _out_path(path, "denoised", output_path)
    _save(img, out)
    return _result(img, out)


@_tool
def upscale(path: str, scale: int = 4, style: str = "photo", output_path: str | None = None):
    """Aumenta la resolución con Real-ESRGAN (en la GPU vía Vulkan).

    scale: 2, 3 o 4. style: "photo" o "anime" (ilustraciones/dibujos).
    """
    scale = int(scale)
    if scale not in (2, 3, 4):
        raise ValueError("scale debe ser 2, 3 o 4")
    out = _out_path(path, f"x{scale}", output_path, ".png")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.png"
        _save(_load(path), src)
        if style == "anime":
            model, s = f"realesr-animevideov3-x{scale}", scale
        else:
            model, s = "realesrgan-x4plus", 4
        tmp = Path(td) / "out.png"
        r = subprocess.run([str(REALESRGAN), "-i", str(src), "-o", str(tmp), "-n", model, "-s", str(s)],
                           cwd=REALESRGAN.parent, capture_output=True, text=True)
        if r.returncode != 0 or not tmp.exists():
            raise RuntimeError(f"Real-ESRGAN falló: {r.stderr[-500:]}")
        img = cv2.imread(str(tmp))
        if s != scale:
            h, w = _load(path).shape[:2]
            img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_AREA)
    _save(img, out)
    return _result(img, out)


@_tool
def remove_background(path: str, background: str | None = None, output_path: str | None = None):
    """Quita el fondo (rembg/ISNet). Sin background deja PNG transparente;
    background acepta un color hex como "#ffffff"."""
    from rembg import new_session, remove
    with _load_lock:
        if "isnet" not in _rembg_sessions:
            _rembg_sessions["isnet"] = new_session("isnet-general-use")
    rgb = cv2.cvtColor(_load(path), cv2.COLOR_BGR2RGB)
    rgba = np.asarray(remove(PILImage.fromarray(rgb), session=_rembg_sessions["isnet"]))
    img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    if background:
        hx = background.lstrip("#")
        color = np.array([int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16)], np.float32)
        alpha = img[..., 3:4].astype(np.float32) / 255
        img = (img[..., :3] * alpha + color * (1 - alpha)).astype(np.uint8)
    out = _out_path(path, "nobg", output_path, None if background else ".png")
    _save(img, out)
    return _result(img, out)


def _lama(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpainting con LaMa sobre un recorte alrededor de la máscara."""
    H, W = img.shape[:2]
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(x1 - x0, y1 - y0) * 2.0) + 64
    side = min(max(side, 256), max(H, W))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    bx0 = int(np.clip(cx - side // 2, 0, max(W - side, 0)))
    by0 = int(np.clip(cy - side // 2, 0, max(H - side, 0)))
    crop = img[by0:by0 + side, bx0:bx0 + side]
    cmask = mask[by0:by0 + side, bx0:bx0 + side]
    ch, cw = crop.shape[:2]
    im512 = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)
    m512 = (cv2.resize(cmask, (512, 512), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
    x = im512.transpose(2, 0, 1)[None].astype(np.float32) / 255
    res = _model("lama_fp32.onnx")([x, m512[None, None]])
    o = next(iter(res.values()))[0].transpose(1, 2, 0)
    if o.max() <= 1.5:
        o = o * 255
    o = cv2.cvtColor(np.clip(o, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    o = cv2.resize(o, (cw, ch), interpolation=cv2.INTER_CUBIC)
    soft = cv2.GaussianBlur(cmask, (0, 0), 3).astype(np.float32)[..., None] / 255
    out = img.copy()
    out[by0:by0 + ch, bx0:bx0 + cw] = (o * soft + crop * (1 - soft)).astype(np.uint8)
    return out


@_tool
def remove_objects(path: str, boxes: list[list[int]] | None = None, mask_path: str | None = None,
                   output_path: str | None = None):
    """Borra objetos y rellena el hueco de forma natural (LaMa).

    boxes: rectángulos [x, y, ancho, alto] en píxeles de la imagen original
    (usá image_info para ver tamaño). O mask_path: imagen blanca donde borrar.
    """
    img = _load(path)
    H, W = img.shape[:2]
    if mask_path:
        mask = cv2.imread(str(Path(mask_path).expanduser()), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        groups = [mask]
    elif boxes:
        groups = []
        for x, y, w, h in boxes:
            m = np.zeros((H, W), np.uint8)
            cv2.rectangle(m, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
            groups.append(m)
    else:
        raise ValueError("Indicá boxes o mask_path")
    for m in groups:
        m = cv2.dilate(m, np.ones((15, 15), np.uint8))
        if m.any():
            img = _lama(img, m)
    out = _out_path(path, "clean", output_path)
    _save(img, out)
    return _result(img, out)


_FFHQ = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                  [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)


@_tool
def restore_faces(path: str, fidelity: float = 0.7, output_path: str | None = None):
    """Restaura caras borrosas, pixeladas o de fotos viejas (GFPGAN 1.4).

    fidelity: 0..1, mezcla con la cara original (1 = solo resultado de IA).
    """
    img = _load(path)
    H, W = img.shape[:2]
    det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (W, H), 0.6)
    _, faces = det.detect(img)
    found = 0 if faces is None else len(faces)
    # con menos de ~40 px el modelo inventa rasgos; mejor reescalar antes
    faces = [] if faces is None else [f for f in faces if min(f[2], f[3]) >= 40]
    if len(faces) == 0:
        out = _out_path(path, "faces", output_path)
        _save(img, out)
        why = (f"Las {found} caras detectadas son demasiado chicas (<40 px); usá upscale primero."
               if found else "No se detectaron caras")
        return _result(img, out, f"{why}; la imagen queda igual.")
    net = _model("gfpgan_1.4.onnx")
    k = float(np.clip(fidelity, 0, 1))
    result = img.astype(np.float32)
    for f in faces:
        lm = f[4:14].reshape(5, 2).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(lm, _FFHQ, method=cv2.LMEDS)
        face = cv2.warpAffine(img, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        x = (cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - 0.5) / 0.5
        o = next(iter(net([x.transpose(2, 0, 1)[None]]).values()))[0].transpose(1, 2, 0)
        o = cv2.cvtColor(np.clip((o + 1) * 127.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        o = cv2.addWeighted(o, k, face, 1 - k, 0)
        Minv = cv2.invertAffineTransform(M)
        back = cv2.warpAffine(o, Minv, (W, H), flags=cv2.INTER_LINEAR)
        m = np.zeros((512, 512), np.float32)
        cv2.ellipse(m, (256, 290), (190, 230), 0, 0, 360, 1, -1)
        m = cv2.GaussianBlur(m, (0, 0), 20)
        mb = cv2.warpAffine(m, Minv, (W, H))[..., None]
        result = back * mb + result * (1 - mb)
    img = np.clip(result, 0, 255).astype(np.uint8)
    out = _out_path(path, "faces", output_path)
    _save(img, out)
    return _result(img, out, f"Caras restauradas: {len(faces)}.")


@_tool
def transform(path: str, crop: list[int] | None = None, width: int | None = None,
              height: int | None = None, rotate: float = 0, flip: str | None = None,
              format: str | None = None, output_path: str | None = None):
    """Recortar, redimensionar, rotar, voltear o convertir formato.

    crop: [x, y, ancho, alto]. width/height: si das solo uno se mantiene la
    proporción. rotate: grados antihorario. flip: "h" o "v".
    format: "jpg", "png" o "webp".
    """
    img = _load(path)
    if crop:
        x, y, w, h = map(int, crop)
        img = img[y:y + h, x:x + w]
    if width or height:
        h0, w0 = img.shape[:2]
        w = width or round(w0 * height / h0)
        h = height or round(h0 * width / w0)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if w < w0 else cv2.INTER_LANCZOS4)
    if rotate:
        if rotate % 90 == 0:
            for _ in range(int(rotate // 90) % 4):
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate, 1)
            c, s = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(h * s + w * c), int(h * c + w * s)
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    if flip:
        img = cv2.flip(img, 1 if flip == "h" else 0)
    ext = f".{format.lower().replace('jpeg', 'jpg')}" if format else None
    out = _out_path(path, "edited", output_path, ext)
    _save(img, out)
    return _result(img, out)


# ---------- comparación ----------

def _label(img: np.ndarray, text: str) -> np.ndarray:
    scale = max(img.shape[1] / 900, 0.6)
    th = max(int(scale * 2), 1)
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    pad = int(10 * scale)
    cv2.rectangle(img, (0, 0), (tw + 2 * pad, tht + 2 * pad), (0, 0, 0), -1)
    cv2.putText(img, text, (pad, tht + pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)
    return img


def _stats(img: np.ndarray) -> tuple[float, float, float]:
    """Brillo medio, saturación media y nitidez (varianza del Laplaciano) a 1500 px."""
    small = cv2.resize(img, (1500, round(1500 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1]
    return float(gray.mean()), float(sat.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


@_tool
def compare(original: str, edited: str, mode: str = "side", region: list[int] | None = None,
            output_path: str | None = None):
    """Compara una foto original con su versión editada.

    mode: "side" (lado a lado) o "split" (mitad izquierda original, derecha editada).
    region: [x, y, ancho, alto] en píxeles del original para ver un detalle al 100%.
    output_path: si se da, guarda la imagen de comparación.
    Devuelve las diferencias de brillo, saturación y nitidez, y una vista previa.
    """
    a, b = _load(original), _load(edited)
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    (ba, sa, na), (bb, sb, nb) = _stats(a), _stats(b)
    if region:
        x, y, w, h = map(int, region)
        a, b = a[y:y + h, x:x + w], b[y:y + h, x:x + w]
    if mode == "split":
        h, w = a.shape[:2]
        img = np.concatenate([a[:, :w // 2], b[:, w // 2:]], axis=1)
        cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), max(w // 400, 2))
        img = _label(img, "Original | Editada")
    else:
        gap = np.full((a.shape[0], max(a.shape[1] // 100, 4), 3), 255, np.uint8)
        img = np.concatenate([_label(a.copy(), "Original"), gap, _label(b.copy(), "Editada")], axis=1)
    msg = (f"Brillo {ba:.0f} -> {bb:.0f} ({bb - ba:+.0f}), saturación {sa:.0f} -> {sb:.0f} ({sb - sa:+.0f}), "
           f"nitidez {na:.0f} -> {nb:.0f} ({(nb / max(na, 1e-6) - 1) * 100:+.0f}%).")
    if output_path:
        out = _out_path(original, "compare", output_path)
        _save(img, out)
        msg = f"Guardado en {out}. " + msg
    return [msg, _preview(img, 1400)]


def _thumb(src: Path, out: Path, max_side: int) -> None:
    with PILImage.open(src) as im:
        im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        im.save(out, "JPEG", quality=85)


_GALLERY = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Antes y después</title>
<style>
body{margin:0;background:#111;color:#eee;font:14px system-ui,sans-serif;display:flex;height:100vh}
#list{width:180px;overflow-y:auto;border-right:1px solid #333;flex-shrink:0}
#list div{padding:6px 10px;cursor:pointer}#list div.on{background:#2a5bd7}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#bar{padding:8px 12px;display:flex;gap:12px;align-items:center;border-bottom:1px solid #333}
#view{flex:1;position:relative;overflow:hidden;cursor:ew-resize;user-select:none}
#view img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
#after{clip-path:inset(0 0 0 50%)}
#line{position:absolute;top:0;bottom:0;width:2px;background:#fff;left:50%;pointer-events:none}
.tag{position:absolute;top:8px;background:#000a;padding:3px 8px;border-radius:4px}
#err{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);max-width:80%;background:#a22;padding:12px 16px;border-radius:6px;display:none}
@media(max-width:700px){body{flex-direction:column}#list{width:auto;height:90px;border:0;border-bottom:1px solid #333}}
</style></head><body>
<div id="list"></div>
<div id="main"><div id="bar"><b id="name"></b><span>← → cambia de foto · arrastrá para comparar · espacio alterna</span></div>
<div id="view"><img id="before"><img id="after"><div id="line"></div>
<span class="tag" style="left:8px">Original</span><span class="tag" style="right:8px">Editada</span><div id="err"></div></div></div>
<script>
const P=__PAIRS__;let i=0,x=50;
const $=id=>document.getElementById(id),L=$('list');
P.forEach((p,k)=>{const d=document.createElement('div');d.textContent=p;d.onclick=()=>show(k);L.append(d)});
function set(v){x=Math.max(0,Math.min(100,v));$('after').style.clipPath=`inset(0 0 0 ${x}%)`;$('line').style.left=x+'%'}
['before','after'].forEach(id=>{$(id).onerror=e=>{const m=$('err');m.style.display='block';
m.textContent=`No se pudo cargar ${e.target.src}. Abrí este index.html directamente desde su carpeta (que contiene antes/ y despues/), en un navegador.`};
$(id).onload=()=>$('err').style.display='none'});
function show(k){i=(k+P.length)%P.length;$('before').src='antes/'+P[i];$('after').src='despues/'+P[i];$('name').textContent=`${P[i]} (${i+1}/${P.length})`;
[...L.children].forEach((d,j)=>d.classList.toggle('on',j==i));L.children[i].scrollIntoView({block:'nearest'})}
const v=$('view');let drag=false;
v.onpointerdown=e=>{drag=true;v.setPointerCapture(e.pointerId);mv(e)};v.onpointerup=()=>drag=false;
v.onpointermove=e=>drag&&mv(e);function mv(e){const r=v.getBoundingClientRect();set((e.clientX-r.left)/r.width*100)}
onkeydown=e=>{if(e.key=='ArrowRight')show(i+1);if(e.key=='ArrowLeft')show(i-1);if(e.key==' '){e.preventDefault();set(x>50?0:100)}};
show(0);
</script></body></html>"""


@_tool
def compare_gallery(original_dir: str, edited_dir: str, output_dir: str | None = None,
                    max_side: int = 1600):
    """Crea una página HTML para recorrer y comparar todas las fotos originales con sus
    versiones editadas (mismo nombre de archivo) con un deslizador antes/después.

    output_dir: carpeta donde se escribe index.html y las miniaturas
    (por defecto <edited_dir>/comparacion). max_side: tamaño de las miniaturas.
    """
    src, dst = Path(original_dir).expanduser(), Path(edited_dir).expanduser()
    exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    names = sorted(f.name for f in dst.iterdir() if f.suffix.lower() in exts and (src / f.name).is_file())
    if not names:
        raise FileNotFoundError(f"No hay fotos con el mismo nombre en {src} y {dst}")
    out = Path(output_dir).expanduser() if output_dir else dst / "comparacion"
    for sub in ("antes", "despues"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for n in names:
        thumb = Path(n).with_suffix(".jpg").name
        _thumb(src / n, out / "antes" / thumb, max_side)
        _thumb(dst / n, out / "despues" / thumb, max_side)
    pairs = [Path(n).with_suffix(".jpg").name for n in names]
    (out / "index.html").write_text(_GALLERY.replace("__PAIRS__", json.dumps(pairs)), encoding="utf-8")
    return f"Galería con {len(names)} pares en {out / 'index.html'}"


# ---------- visor interactivo ----------

_PARAMS = {"strength": (0.6, 0, 2), "brightness": (0, -100, 100), "contrast": (0, -100, 100),
           "saturation": (0, -100, 100), "warmth": (0, -100, 100), "gamma": (1.0, 0.3, 3),
           "sharpness": (0, 0, 100), "scene_amount": (1.0, 0, 1.5)}
_viewer = {"httpd": None, "src": None, "dst": None, "port": None}
_bases: OrderedDict = OrderedDict()
_bases_lock = threading.Lock()


def _params(raw: dict) -> dict:
    out = {}
    for k, (default, lo, hi) in _PARAMS.items():
        try:
            out[k] = float(np.clip(float(raw.get(k, default)), lo, hi))
        except (TypeError, ValueError):
            out[k] = default
    scene = str(raw.get("scene") or "")
    out["scene"] = scene if scene == "auto" or scene in _scenes().PROFILES else ""
    return out


def _scenes():
    import scenes
    return scenes


_scene_cache: OrderedDict = OrderedDict()
_scene_lock = threading.Lock()


def _scene_for(path) -> tuple[str, dict]:
    """Escena detectada de una foto. Siempre sobre la misma versión (1600 px + EXIF) y en
    caché por archivo, para que la vista previa, el guardado y el MCP usen la misma escena."""
    path = Path(path).expanduser()
    key = (str(path), path.stat().st_mtime)
    with _scene_lock:
        if key not in _scene_cache:
            sc = _scenes()
            _scene_cache[key] = sc.detect_scene(_base(path, 1600)[0], sc.read_exif(path))
            while len(_scene_cache) > 256:
                _scene_cache.popitem(last=False)
        return _scene_cache[key]


def _check_scene(scene: str) -> str:
    if scene and scene != "auto" and scene not in _scenes().PROFILES:
        raise ValueError(f"Escena desconocida '{scene}'. Opciones: auto, {', '.join(_scenes().PROFILES)}")
    return scene


def _edit(img: np.ndarray, p: dict, scale: float = 1.0, path=None) -> np.ndarray:
    """Aplica los ajustes p. path hace falta solo para resolver scene="auto"."""
    if p.get("scene"):
        scene = _scene_for(path)[0] if p["scene"] == "auto" else p["scene"]
        img = _scenes().enhance(img, scene, p.get("scene_amount", 1.0), scale)
    if p["strength"] > 0:
        img = _enhance(img, p["strength"], scale)
    return _adjust(img, p["brightness"], p["contrast"], p["saturation"], p["warmth"],
                   p["gamma"], p["sharpness"], scale)


def _base(path: Path, max_side: int) -> tuple[np.ndarray, float]:
    """Versión reducida de la foto (en caché) y su escala respecto del tamaño completo."""
    key = (str(path), path.stat().st_mtime, max_side)
    with _bases_lock:
        if key in _bases:
            _bases.move_to_end(key)
            return _bases[key]
    with PILImage.open(path) as im:
        full = max(im.size)
        im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    entry = (img, max(img.shape[:2]) / full)
    with _bases_lock:
        _bases[key] = entry
        while len(_bases) > 16:
            _bases.popitem(last=False)
    return entry


def _viewer_names() -> list[str]:
    src, dst = _viewer["src"], _viewer["dst"]
    exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    return sorted(f.name for f in src.iterdir() if f.suffix.lower() in exts)


def _full_width(path: Path) -> int:
    with PILImage.open(path) as im:
        w, h = im.size
        return h if im.getexif().get(0x0112) in (5, 6, 7, 8) else w


def _saved_params() -> dict:
    f = _viewer["dst"] / "ajustes.json"
    return json.loads(f.read_text()) if f.is_file() else {}


# ---------- ajustes con nombre y aplicación en lote ----------

PRESETS = KIT / "presets.json"
_ajustes_lock = threading.Lock()
_job = {"running": False, "total": 0, "done": 0, "saved": 0, "errors": [], "cancel": False, "names": []}


def _presets() -> dict:
    return json.loads(PRESETS.read_text()) if PRESETS.is_file() else {}


def _store_preset(name: str, params: dict | None) -> dict:
    """Guarda (o borra, con params=None) un ajuste con nombre."""
    name = name.strip()
    if not name:
        raise ValueError("El ajuste necesita un nombre")
    presets = _presets()
    if params is None:
        presets.pop(name, None)
    else:
        presets[name] = _params(params)
    PRESETS.write_text(json.dumps(presets, indent=1, ensure_ascii=False))
    return presets


def _apply_one(src: Path, dst: Path, name: str, p: dict) -> None:
    _save(_edit(_load(str(src / name)), p, path=src / name), dst / name)
    _trim_memory()
    with _ajustes_lock:
        f = dst / "ajustes.json"
        saved = json.loads(f.read_text()) if f.is_file() else {}
        saved[name] = p
        f.write_text(json.dumps(saved, indent=1))


def _apply_many(src: Path, dst: Path, names: list[str], p: dict, job: dict | None = None) -> list[str]:
    """Aplica p a cada foto a resolución completa (3 a la vez). Devuelve los errores."""
    errors = []

    def one(n):
        if job and job["cancel"]:
            return False
        _apply_one(src, dst, n, p)
        return True

    # un perfil de escena usa ~4 GB por foto de 24 MP (y ya usa varios núcleos): de a una
    with ThreadPoolExecutor(max_workers=1 if p.get("scene") else 3) as ex:
        futures = {ex.submit(one, n): n for n in names}
        for f in as_completed(futures):
            saved = False
            try:
                saved = f.result()
            except Exception as e:
                errors.append(f"{futures[f]}: {e}")
            if job:
                job["done"] += 1
                job["saved"] += saved
                job["errors"] = errors
    return errors


def _start_job(names: list[str], p: dict) -> None:
    if _job["running"]:
        raise RuntimeError("Ya se está aplicando un ajuste a varias fotos")
    _job.update(running=True, total=len(names), done=0, saved=0, errors=[], cancel=False, names=names)
    src, dst = _viewer["src"], _viewer["dst"]

    def run():
        try:
            _apply_many(src, dst, names, p, _job)
        finally:
            _job["running"] = False

    threading.Thread(target=run, daemon=True).start()


class _ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # stdout/stderr son del protocolo MCP
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def _jpeg(self, img: np.ndarray):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        self._send(buf.tobytes(), "image/jpeg")

    def _name(self, name: str) -> str:
        if name not in _viewer_names():
            raise FileNotFoundError(name)
        return name

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                self._send(_VIEWER.encode(), "text/html; charset=utf-8")
            elif u.path == "/api/info":
                self._json({"width": _full_width(_viewer["src"] / self._name(q.get("name", "")))})
            elif u.path == "/api/list":
                dst = _viewer["dst"]
                saved = _saved_params()
                self._json({"photos": [{"name": n, "edited": (dst / n).is_file(), "params": saved.get(n)}
                                       for n in _viewer_names()],
                            "defaults": {**{k: v[0] for k, v in _PARAMS.items()}, "scene": ""},
                            "scenes": {k: v["label"] for k, v in _scenes().PROFILES.items()},
                            "src": str(_viewer["src"]), "dst": str(dst)})
            elif u.path.startswith("/img/"):
                kind, name = unquote(u.path[5:]).split("/", 1)
                folder = _viewer["src"] if kind == "original" else _viewer["dst"]
                w = int(np.clip(int(q.get("w", 1600)), 200, 8000))
                self._jpeg(_base(folder / self._name(name), w)[0])
            elif u.path == "/preview":
                w = int(np.clip(int(q.get("w", 1600)), 200, 8000))
                path = _viewer["src"] / self._name(q.get("name", ""))
                img, scale = _base(path, w)
                self._jpeg(_edit(img, _params(q), scale, path))
            elif u.path == "/api/scene":
                path = _viewer["src"] / self._name(q.get("name", ""))
                scene, probs = _scene_for(path)
                self._json({"scene": scene, "label": _scenes().PROFILES[scene]["label"], "probs": probs})
            elif u.path == "/api/presets":
                self._json(_presets())
            elif u.path == "/api/job":
                self._json({**_job, "cancelled": _job["cancel"]})
            else:
                self._json({"error": "no encontrado"}, 404)
        except FileNotFoundError as e:
            self._json({"error": f"no existe: {e}"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/save":
                name, p = self._name(body.get("name", "")), _params(body.get("params", {}))
                _apply_one(_viewer["src"], _viewer["dst"], name, p)
                self._json({"ok": True, "params": p})
            elif path == "/api/apply":
                names = [self._name(n) for n in body.get("names", [])]
                if not names:
                    return self._json({"error": "No hay fotos seleccionadas"}, 400)
                _start_job(names, _params(body.get("params", {})))
                self._json({"ok": True, "total": len(names)})
            elif path == "/api/cancel":
                _job["cancel"] = True
                self._json({"ok": True})
            elif path == "/api/presets":
                self._json(_store_preset(body.get("name", ""), None if body.get("delete") else body.get("params", {})))
            else:
                self._json({"error": "no encontrado"}, 404)
        except FileNotFoundError as e:
            self._json({"error": f"no existe: {e}"}, 404)
        except (ValueError, RuntimeError) as e:
            self._json({"error": str(e)}, 409)
        except Exception as e:
            self._json({"error": str(e)}, 500)


@_tool
def open_viewer(original_dir: str, edited_dir: str, port: int = 8765, open_browser: bool = True):
    """Abre un visor web local para comparar originales con editadas (mismo nombre de
    archivo) con deslizador antes/después, zoom, paneo e histograma, y ajustar en vivo el
    perfil de escena, la mejora automática, brillo, contraste, saturación, calidez, gamma y
    nitidez (siempre a partir del original, sin acumular ediciones). "Guardar" aplica
    los ajustes a la foto completa y la escribe en edited_dir (los originales no se tocan);
    los ajustes quedan en edited_dir/ajustes.json. Devuelve la URL.
    """
    src, dst = Path(original_dir).expanduser(), Path(edited_dir).expanduser()
    if not src.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    _viewer.update(src=src, dst=dst)
    if _viewer["httpd"] is None:
        for p in range(port, port + 20):  # otro visor puede estar usando el puerto
            try:
                httpd = ThreadingHTTPServer(("127.0.0.1", p), _ViewerHandler)
                break
            except OSError:
                continue
        else:
            raise OSError(f"No hay puertos libres entre {port} y {port + 19}")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        _viewer.update(httpd=httpd, port=p)
    url = f"http://127.0.0.1:{_viewer['port']}/"
    if open_browser:
        webbrowser.open(url)
    return f"Visor abierto en {url} con {len(_viewer_names())} fotos de {src} (editadas en {dst})."


@_tool
def list_presets():
    """Lista los ajustes guardados con nombre (los mismos que usa el visor)."""
    presets = _presets()
    if not presets:
        return "No hay ajustes guardados."
    fmt = lambda v: f"{v:g}" if isinstance(v, (int, float)) else (v or "ninguna")
    return "\n".join(f"{n}: " + ", ".join(f"{k}={fmt(v)}" for k, v in p.items()) for n, p in presets.items())


@_tool
def save_preset(name: str, strength: float | None = None, brightness: float = 0, contrast: float = 0,
                saturation: float = 0, warmth: float = 0, gamma: float = 1.0, sharpness: float = 0,
                scene: str = "", scene_amount: float = 1.0):
    """Guarda un ajuste con nombre para aplicarlo después a varias fotos.
    strength es la mejora automática clásica (0..2); por defecto 0.6, o 0 si hay escena
    (el perfil ya incluye esa mejora, igual que en el visor). El resto como en adjust.
    scene: perfil de escena ("auto", "nublado", "noche", … ver list_scenes) o "" para ninguno;
    se aplica antes que los demás ajustes, con intensidad scene_amount (0..1.5)."""
    _check_scene(scene)
    if strength is None:
        strength = 0 if scene else 0.6
    p = _store_preset(name, dict(strength=strength, brightness=brightness, contrast=contrast,
                                 saturation=saturation, warmth=warmth, gamma=gamma, sharpness=sharpness,
                                 scene=scene, scene_amount=scene_amount))
    return f"Ajuste '{name.strip()}' guardado. Hay {len(p)} ajustes guardados."


@_tool
def apply_preset(original_dir: str, edited_dir: str, preset: str, files: list[str] | None = None):
    """Aplica un ajuste guardado a varias fotos de original_dir y escribe el resultado
    con el mismo nombre en edited_dir (los originales no se tocan).
    files: nombres de archivo; si se omite, se aplica a todas las fotos de la carpeta.
    """
    presets = _presets()
    if preset not in presets:
        raise ValueError(f"No existe el ajuste '{preset}'. Disponibles: {', '.join(presets) or 'ninguno'}")
    src, dst = Path(original_dir).expanduser(), Path(edited_dir).expanduser()
    exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    available = sorted(f.name for f in src.iterdir() if f.suffix.lower() in exts)
    names = available if files is None else [Path(f).name for f in files]
    missing = [n for n in names if n not in available]
    if missing:
        raise FileNotFoundError(f"No están en {src}: {', '.join(missing)}")
    dst.mkdir(parents=True, exist_ok=True)
    errors = _apply_many(src, dst, names, _params(presets[preset]))
    msg = f"Ajuste '{preset}' aplicado a {len(names) - len(errors)} de {len(names)} fotos en {dst}."
    return msg + ("\nErrores:\n" + "\n".join(errors) if errors else "")


_VIEWER = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Visor antes y después</title>
<style>
:root{--bg:#111;--panel:#1b1b1b;--line:#333;--fg:#eee;--dim:#999;--acc:#2a5bd7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}
#list{width:200px;border-right:1px solid var(--line);flex-shrink:0;display:flex;flex-direction:column;min-height:0}
#listbar{padding:7px 8px;border-bottom:1px solid var(--line);display:flex;gap:4px;flex-wrap:wrap;align-items:center}
#listbar button{padding:3px 7px}#selcount{color:var(--dim);width:100%}
#items{overflow-y:auto;flex:1}
#items div{padding:4px 8px;cursor:pointer;white-space:nowrap;display:flex;gap:6px;align-items:center}
#items div.on{background:var(--acc)}#items div.saved::after{content:"●";color:#6c6;margin-left:auto}
#items input{margin:0;cursor:pointer}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#bar{padding:7px 12px;display:flex;gap:12px;align-items:center;border-bottom:1px solid var(--line);flex-wrap:wrap}
#bar .hint{color:var(--dim)}
#stage{flex:1;position:relative;overflow:hidden;cursor:grab;touch-action:none;user-select:none}
#stage.panning{cursor:grabbing}
#canvas{position:absolute;left:0;top:0;transform-origin:0 0}
#canvas img{position:absolute;inset:0;width:100%;height:100%;display:block;-webkit-user-drag:none}
#line{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;background:#fff;pointer-events:none;box-shadow:0 0 4px #000}
#handle{position:absolute;top:50%;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;background:#fff;
 color:#111;display:flex;align-items:center;justify-content:center;cursor:ew-resize;font-size:14px;box-shadow:0 0 6px #000}
.tag{position:absolute;top:8px;background:#000b;padding:3px 8px;border-radius:4px;pointer-events:none}
#err{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);max-width:80%;background:#a22;padding:12px 16px;border-radius:6px;display:none}
#busy{position:absolute;right:8px;bottom:8px;background:#000b;padding:3px 8px;border-radius:4px;display:none}
#side{width:250px;flex-shrink:0;border-left:1px solid var(--line);background:var(--panel);padding:12px;overflow-y:auto}
#side h3{margin:0 0 10px;font-size:13px}
.ctl{margin-bottom:12px}.ctl label{display:flex;justify-content:space-between;margin-bottom:3px}
.ctl label span{color:var(--dim);font-variant-numeric:tabular-nums}
.ctl input{width:100%}
button{background:#333;color:var(--fg);border:1px solid #444;border-radius:5px;padding:6px 10px;cursor:pointer;font:inherit}
button:hover{background:#3d3d3d}button.pri{background:var(--acc);border-color:var(--acc)}
.row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
#msg{color:var(--dim);min-height:1.2em;margin-top:6px}
#side h3.sec{margin-top:18px;padding-top:12px;border-top:1px solid var(--line)}
select{width:100%;background:#2a2a2a;color:var(--fg);border:1px solid #444;border-radius:5px;padding:5px;font:inherit;margin-bottom:8px}
progress{width:100%;margin:6px 0}#jobmsg{color:var(--dim);min-height:1.2em}
#hist{width:100%;height:120px;display:block;background:#0b0b0b;border:1px solid var(--line);border-radius:4px}
.seg{display:flex;margin-bottom:6px}.seg button{flex:1;border-radius:0;padding:3px 4px}
.seg button:first-child{border-radius:5px 0 0 5px}.seg button:last-child{border-radius:0 5px 5px 0}
.seg button.on{background:var(--acc);border-color:var(--acc)}
#hstats{color:var(--dim);margin:5px 0 14px;line-height:1.5;font-variant-numeric:tabular-nums}
#hstats b{color:#e66;font-weight:normal}
#scenemsg{color:var(--dim);margin:-2px 0 14px;min-height:1.2em}
@media(max-width:800px){body{flex-direction:column;overflow:auto;height:auto}
 #list{width:auto;height:160px;border:0;border-bottom:1px solid var(--line)}
 #stage{height:60vh;flex:none}#side{width:auto;border:0}}
</style></head><body>
<div id="list"><div id="listbar"><button id="selall">Todas</button><button id="selnone">Ninguna</button>
 <button id="selinv">Invertir</button><span id="selcount"></span></div><div id="items"></div></div>
<div id="main">
 <div id="bar"><b id="name"></b><span id="zoom">100%</span><button id="fit">Ajustar</button><button id="z100">1:1</button>
  <span class="hint">rueda: zoom · arrastrar: mover · círculo: comparar · ← →: foto · espacio: alternar · 0: ajustar</span></div>
 <div id="stage"><div id="canvas"><img id="before" alt=""><img id="after" alt=""></div>
  <div id="line"></div><div id="handle">⇔</div>
  <span class="tag" style="left:8px">Original</span><span class="tag" id="rtag" style="right:8px">Editada</span>
  <div id="err"></div><div id="busy">procesando…</div></div>
</div>
<div id="side"><h3>Histograma</h3>
 <div class="seg"><button id="h_after" class="on">Editada</button><button id="h_before">Original</button><button id="h_both">Ambas</button></div>
 <canvas id="hist" title="Izquierda: sombras · derecha: luces. Rojo, verde y azul por canal; blanco: luminosidad."></canvas>
 <div id="hstats"></div>
 <h3>Perfil de escena</h3>
 <select id="scene"><option value="">Sin perfil</option><option value="auto">Automático (detectar escena)</option></select>
 <div id="scenemsg"></div>
 <h3>Ajustes</h3><div id="ctls"></div>
 <div class="row"><button class="pri" id="save">Guardar</button><button id="reset">Restablecer</button></div>
 <div class="row"><button id="copy">Copiar ajustes</button><button id="paste">Pegar</button></div>
 <div id="msg"></div>
 <h3 class="sec">Ajustes guardados</h3>
 <select id="preset"><option value="">— elegí uno para cargarlo —</option></select>
 <div class="row"><button id="psave">Guardar como…</button><button id="pdel">Borrar</button></div>
 <h3 class="sec">Aplicar a varias fotos</h3>
 <div class="row"><button class="pri" id="apply">Aplicar a seleccionadas</button><button id="cancel" hidden>Cancelar</button></div>
 <progress id="prog" value="0" max="1" hidden></progress><div id="jobmsg">Marcá fotos en la lista (Mayús+clic marca un rango).</div></div>
<script>
const $=id=>document.getElementById(id);
const CTL=[["strength","Mejora automática",0,2,.05],["brightness","Brillo",-100,100,1],["contrast","Contraste",-100,100,1],
 ["saturation","Saturación",-100,100,1],["warmth","Calidez",-100,100,1],["gamma","Gamma (sombras)",.3,3,.05],["sharpness","Nitidez",0,100,1],
 ["scene_amount","Intensidad del perfil",0,1.5,.05]];
let photos=[],defs={},i=0,p={},dirty=false,clip=null,ver=0,presets={},sel=new Set(),lastCheck=-1;
let z=1,tx=0,ty=0,d=.5,fw=1,fh=1,ar=1.5,res=0,fullW=6000,mode=null,last=null,pTimer=0,rTimer=0;
let busy=false,again=false,shown='',hiTimer=0,lastInput=0;
const S=$('stage'),C=$('canvas'),B=$('before'),A=$('after');

CTL.forEach(([k,l,lo,hi,st])=>{const c=document.createElement('div');c.className='ctl';
 c.innerHTML=`<label for="c_${k}">${l}<span id="v_${k}"></span></label><input type="range" id="c_${k}" min="${lo}" max="${hi}" step="${st}">`;
 $('ctls').append(c);const r=c.querySelector('input');
 r.oninput=()=>{p[k]=+r.value;$('v_'+k).textContent=r.value;pickPreset();dirty=true;lastInput=performance.now();schedule()};
 r.ondblclick=()=>{p[k]=defs[k];sync();dirty=true;schedule()}});
function sync(){CTL.forEach(([k])=>{$('c_'+k).value=p[k];$('v_'+k).textContent=p[k]});$('scene').value=p.scene||'';
 $('c_scene_amount').closest('.ctl').style.opacity=p.scene?1:.4;pickPreset();if(photos.length)showScene()}
// el menú muestra el ajuste guardado que coincide con los controles, o queda en blanco
function pickPreset(){const n=Object.keys(presets).find(n=>{const q={...defs,...presets[n]};
 return CTL.every(([k])=>+q[k]==+p[k])&&(q.scene||'')==(p.scene||'')});$('preset').value=n||''}
// perfil de escena: en "auto" se muestra qué escena detectó el servidor
let sceneLabels={};
function showScene(){const m=$('scenemsg'),at=i;
 if(!p.scene)return m.textContent='';if(p.scene!='auto')return m.textContent='Perfil aplicado antes de los ajustes.';
 m.textContent='Detectando escena…';
 fetch('/api/scene?name='+encodeURIComponent(photos[i].name)).then(r=>r.json()).then(j=>{if(at!=i||p.scene!='auto')return;
  const pr=Object.entries(j.probs||{}).slice(0,3).map(([k,v])=>`${sceneLabels[k]||k} ${Math.round(v*100)}%`).join(' · ');
  m.textContent=`Detectada: ${j.label}. ${pr}`}).catch(()=>m.textContent='No se pudo detectar la escena.')}
$('scene').onchange=()=>{p.scene=$('scene').value;
 if(p.scene&&p.strength>0){p.strength=0;msg('Mejora automática clásica puesta en 0: el perfil ya incluye esa mejora.')}
 sync();dirty=true;schedule()};
const msg=t=>$('msg').textContent=t;

fetch('/api/list').then(r=>r.json()).then(j=>{photos=j.photos;defs=j.defaults;sceneLabels=j.scenes||{};
 for(const [k,l] of Object.entries(sceneLabels))$('scene').add(new Option(l,k));
 if(!photos.length)return err('No hay fotos en '+j.src);
 photos.forEach((ph,k)=>{const e=document.createElement('div'),c=document.createElement('input');c.type='checkbox';
  c.onclick=ev=>{ev.stopPropagation();const on=c.checked;
   const [a,b]=ev.shiftKey&&lastCheck>=0?[Math.min(lastCheck,k),Math.max(lastCheck,k)]:[k,k];
   for(let j=a;j<=b;j++)on?sel.add(j):sel.delete(j);lastCheck=k;marks()};
  e.append(c,document.createTextNode(ph.name));e.onclick=()=>show(k);$('items').append(e)});
 marks();show(0)}).catch(down);
function marks(){[...$('items').children].forEach((e,j)=>{e.classList.toggle('on',j==i);e.classList.toggle('saved',!!photos[j].params);
 e.firstChild.checked=sel.has(j)});$('selcount').textContent=`${sel.size} de ${photos.length} seleccionadas`;
 $('apply').textContent=sel.size?`Aplicar a ${sel.size} seleccionada${sel.size>1?'s':''}`:'Aplicar a seleccionadas'}
$('selall').onclick=()=>{photos.forEach((_,j)=>sel.add(j));marks()};
$('selnone').onclick=()=>{sel.clear();marks()};
$('selinv').onclick=()=>{photos.forEach((_,j)=>sel.has(j)?sel.delete(j):sel.add(j));marks()};

function loadPresets(pick){return fetch('/api/presets').then(r=>r.json()).then(j=>{presets=j;const s=$('preset');
 s.length=1;Object.keys(j).sort().forEach(n=>s.add(new Option(n,n)));if(pick&&j[pick])s.value=pick;else pickPreset()})}
loadPresets();
$('preset').onchange=()=>{const n=$('preset').value;if(!n)return;p={...defs,...presets[n]};sync();dirty=true;schedule();msg(`Cargado "${n}". Guardá esta foto o aplicalo a varias.`)};
async function post(url,body){const r=await fetch(url,{method:'POST',body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw new Error(j.error);return j}
$('psave').onclick=async()=>{const n=prompt('Nombre para estos ajustes:',$('preset').value||'');if(!n||!n.trim())return;
 if(presets[n.trim()]&&!confirm(`Ya existe "${n.trim()}". ¿Reemplazarlo?`))return;
 try{await post('/api/presets',{name:n,params:p});await loadPresets(n.trim());msg(`Ajustes guardados como "${n.trim()}".`)}catch(x){msg('Error: '+x.message)}};
$('pdel').onclick=async()=>{const n=$('preset').value;if(!n)return msg('Elegí primero un ajuste guardado.');
 if(!confirm(`¿Borrar el ajuste "${n}"?`))return;
 try{await post('/api/presets',{name:n,delete:true});await loadPresets();msg(`Ajuste "${n}" borrado.`)}catch(x){msg('Error: '+x.message)}};

$('apply').onclick=async()=>{if(!sel.size)return $('jobmsg').textContent='Primero marcá fotos en la lista.';
 const names=[...sel].sort((a,b)=>a-b).map(j=>photos[j].name),src=$('preset').value;
 if(!confirm(`¿Aplicar ${src?`"${src}"`:'los ajustes actuales'} a ${names.length} foto${names.length>1?'s':''}? `+
  'Se reemplazan sus versiones editadas (los originales no se tocan).'))return;
 try{await post('/api/apply',{names,params:p});watchJob()}catch(x){$('jobmsg').textContent='Error: '+x.message}};
$('cancel').onclick=()=>post('/api/cancel',{}).catch(()=>{});
async function watchJob(){$('apply').disabled=true;$('cancel').hidden=false;$('prog').hidden=false;
 for(;;){let j;try{j=await (await fetch('/api/job')).json()}catch(x){down();break}
  $('prog').max=j.total||1;$('prog').value=j.done;$('jobmsg').textContent=`Procesando ${j.done} de ${j.total}…`;
  if(!j.running){
   $('jobmsg').textContent=(j.cancelled?`Cancelado: se guardaron ${j.saved} de ${j.total}.`:`Listo: ${j.saved} de ${j.total} fotos guardadas.`)+
    (j.errors.length?` ${j.errors.length} con error: `+j.errors.join('; '):'');await refresh(j.names);break}
  await new Promise(x=>setTimeout(x,700))}
 $('apply').disabled=false;$('cancel').hidden=true;$('prog').hidden=true}
async function refresh(names){const j=await (await fetch('/api/list')).json();
 j.photos.forEach((ph,k)=>{if(photos[k]&&photos[k].name==ph.name){photos[k].params=ph.params;photos[k].edited=ph.edited}});
 marks();if(names.includes(photos[i].name)){ver++;p={...defs,...(photos[i].params||{})};sync();dirty=false;shown='';loadAfter()}}
fetch('/api/job').then(r=>r.json()).then(j=>{if(j.running)watchJob()}).catch(()=>{});

function show(k){i=(k+photos.length)%photos.length;const ph=photos[i];
 p={...defs,...(ph.params||{})};sync();dirty=false;res=0;z=1;shown='';A.style.visibility='hidden';
 $('name').textContent=`${ph.name} (${i+1}/${photos.length})`;marks();$('items').children[i].scrollIntoView({block:'nearest'});
 msg(ph.params?'Ajustes guardados con este visor.':ph.edited?'Editada: versión guardada en la carpeta (ajustes desconocidos).':'Sin versión editada todavía.');
 fetch('/api/info?name='+encodeURIComponent(ph.name)).then(r=>r.json()).then(j=>{fullW=j.width||fullW;draw()}).catch(down);
 B.onload=()=>{ar=B.naturalWidth/B.naturalHeight;fit();B.onload=null};load(true)}

function want(){const px=z*fw*devicePixelRatio;return px>4000?8000:px>1700?4000:1600}
function q(){return CTL.map(([k])=>`${k}=${p[k]}`).join('&')+`&scene=${p.scene||''}`}
function load(force){const w=want();if(!force&&w==res)return;res=w;const n=encodeURIComponent(photos[i].name);
 B.src=`/img/original/${n}?w=${w}`;loadAfter()}
// Una sola petición a la vez: mientras se mueve un control se pide una vista rápida
// (máx. 1600 px) y al soltar, la resolución que pide el zoom.
function loadAfter(){if(busy){again=true;return}
 const ph=photos[i],n=encodeURIComponent(ph.name),live=dirty||!ph.edited,at=i;
 const w=live&&performance.now()-lastInput<300?Math.min(res,1600):res;
 const url=live?`/preview?name=${n}&w=${w}&${q()}`:`/img/edited/${n}?w=${w}&v=${ver}`;
 $('rtag').textContent=live?'Vista previa':'Editada';
 if(url==shown)return;busy=true;again=false;$('busy').style.display='block';
 const im=new Image();
 im.onload=()=>{if(at==i){A.src=url;shown=url;A.style.visibility='visible';$('err').style.display='none';hist.after=histo(im);drawHist()}done()};
 // la imagen no dice por qué falló: se repite el pedido para leer el error, o avisar si el visor está apagado
 im.onerror=()=>{if(at==i)fetch(url).then(r=>r.json()).then(j=>err('No se pudo generar la imagen editada: '+(j.error||'error desconocido')))
  .catch(down);done()};
 function done(){busy=false;$('busy').style.display='none';
  if(again||at!=i)loadAfter();else if(w<res){clearTimeout(hiTimer);hiTimer=setTimeout(loadAfter,350)}}
 im.src=url}
function schedule(){clearTimeout(pTimer);pTimer=setTimeout(loadAfter,0)}
function err(t){const e=$('err');e.textContent=t;e.style.display='block'}
function down(){err('No hay conexión con el visor. Volvé a abrirlo (open_viewer o "python server.py viewer …").')}
B.onerror=()=>down();

// ---------- histograma ----------
// Se calcula en el navegador sobre la imagen ya cargada (reducida a ~400 px, alcanza para la forma).
const hist={after:null,before:null,mode:'after'};
B.addEventListener('load',()=>{hist.before=histo(B);if(!A.style.visibility||A.style.visibility=='hidden')hist.after=null;drawHist()});
function histo(img){const k=Math.min(1,400/Math.max(img.naturalWidth,img.naturalHeight)),
 w=Math.max(1,Math.round(img.naturalWidth*k)),h=Math.max(1,Math.round(img.naturalHeight*k));
 const c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d',{willReadFrequently:true});
 x.drawImage(img,0,0,w,h);const d=x.getImageData(0,0,w,h).data,r=new Float32Array(256),g=new Float32Array(256),
 b=new Float32Array(256),l=new Float32Array(256);let sum=0;
 for(let j=0;j<d.length;j+=4){r[d[j]]++;g[d[j+1]]++;b[d[j+2]]++;const y=Math.round(.2126*d[j]+.7152*d[j+1]+.0722*d[j+2]);l[y]++;sum+=y}
 const n=d.length/4,clip=(a,lo,hi)=>{let s=0;for(let v=lo;v<=hi;v++)s+=a[v];return 100*s/n};
 return {r,g,b,l,n,mean:sum/n,dark:clip(l,0,2),light:clip(l,253,255)}}
function drawHist(){const cv=$('hist'),dpr=devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;
 cv.width=W*dpr;cv.height=H*dpr;const x=cv.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,W,H);
 x.strokeStyle='#ffffff14';for(let q=1;q<4;q++){x.beginPath();x.moveTo(W*q/4,0);x.lineTo(W*q/4,H);x.stroke()}
 const main=hist.mode=='before'?hist.before:hist.after||hist.before,ref=hist.mode=='both'?hist.before:null;
 if(!main){$('hstats').textContent='';return}
 // escala con los valores intermedios, para que un pico en negro o blanco puro no aplaste el resto
 let top=1;for(const a of [main.r,main.g,main.b,main.l])for(let v=3;v<253;v++)top=Math.max(top,a[v]);top*=1.08;
 const path=(a,smooth=true)=>{x.beginPath();x.moveTo(0,H);for(let v=0;v<256;v++){
  const y=smooth&&v>0&&v<255?(a[v-1]+2*a[v]+a[v+1])/4:a[v];x.lineTo(v/255*W,H-Math.min(1,y/top)*(H-2))}x.lineTo(W,H);x.closePath()};
 if(ref&&ref!==main){path(ref.l);x.strokeStyle='#ffffff55';x.setLineDash([3,3]);x.stroke();x.setLineDash([])}
 x.globalCompositeOperation='lighter';
 for(const [a,c] of [[main.r,'#e0323299'],[main.g,'#2fbf3f99'],[main.b,'#3a6df099']]){path(a);x.fillStyle=c;x.fill()}
 x.globalCompositeOperation='source-over';path(main.l);x.strokeStyle='#ffffffcc';x.lineWidth=1.2;x.stroke();
 // avisos de recorte en las esquinas
 if(main.dark>0.5){x.fillStyle='#4aa3ff';x.beginPath();x.moveTo(2,2);x.lineTo(12,2);x.lineTo(2,12);x.fill()}
 if(main.light>0.5){x.fillStyle='#ff5a5a';x.beginPath();x.moveTo(W-2,2);x.lineTo(W-12,2);x.lineTo(W-2,12);x.fill()}
 const f=v=>v<0.05?'0%':v<10?v.toFixed(1)+'%':Math.round(v)+'%',wrn=(v,t)=>v>0.5?`<b>${t} ${f(v)}</b>`:`${t} ${f(v)}`;
 const ex=main.mean<70?'oscura':main.mean>185?'muy clara':'equilibrada';
 let s=`Brillo medio ${Math.round(main.mean)} de 255 (${ex})<br>${wrn(main.dark,'Sombras recortadas')} · ${wrn(main.light,'luces recortadas')}`;
 if(hist.mode=='both'&&hist.before&&hist.after)s+=`<br>Original: brillo ${Math.round(hist.before.mean)} · sombras ${f(hist.before.dark)} · luces ${f(hist.before.light)}`;
 $('hstats').innerHTML=s}
for(const m of ['after','before','both'])$('h_'+m).onclick=()=>{hist.mode=m;
 for(const o of ['after','before','both'])$('h_'+o).classList.toggle('on',o==m);drawHist()};
addEventListener('resize',drawHist);

function fit(){const r=S.getBoundingClientRect();fw=Math.min(r.width,r.height*ar);fh=fw/ar;
 C.style.width=fw+'px';C.style.height=fh+'px';z=1;tx=(r.width-fw)/2;ty=(r.height-fh)/2;draw()}
function clamp(){const r=S.getBoundingClientRect(),W=fw*z,H=fh*z;
 tx=W<=r.width?(r.width-W)/2:Math.min(0,Math.max(r.width-W,tx));ty=H<=r.height?(r.height-H)/2:Math.min(0,Math.max(r.height-H,ty))}
function draw(){C.style.transform=`translate(${tx}px,${ty}px) scale(${z})`;const r=S.getBoundingClientRect(),sx=d*r.width;
 const pc=Math.max(0,Math.min(100,(sx-tx)/(fw*z)*100));A.style.clipPath=`inset(0 0 0 ${pc}%)`;
 $('line').style.left=$('handle').style.left=sx+'px';$('zoom').textContent=Math.round(z*fw*devicePixelRatio/fullW*100)+'%';
 clearTimeout(rTimer);rTimer=setTimeout(()=>load(false),250)}
function zoomAt(nz,mx,my){nz=Math.max(1,Math.min(40,nz));tx=mx-(mx-tx)*nz/z;ty=my-(my-ty)*nz/z;z=nz;clamp();draw()}
S.addEventListener('wheel',e=>{e.preventDefault();const r=S.getBoundingClientRect();zoomAt(z*Math.exp(-e.deltaY*.0015),e.clientX-r.left,e.clientY-r.top)},{passive:false});
const pts=new Map();let pinch=null;
S.onpointerdown=e=>{S.setPointerCapture(e.pointerId);pts.set(e.pointerId,{x:e.clientX,y:e.clientY});
 if(pts.size==2){const [a,b]=[...pts.values()];pinch={d:Math.hypot(a.x-b.x,a.y-b.y),z};mode=null;return}
 mode=e.target.id=='handle'?'div':'pan';last={x:e.clientX,y:e.clientY};if(mode=='pan')S.classList.add('panning')};
S.onpointermove=e=>{if(!pts.has(e.pointerId))return;pts.set(e.pointerId,{x:e.clientX,y:e.clientY});const r=S.getBoundingClientRect();
 if(pinch&&pts.size==2){const [a,b]=[...pts.values()];zoomAt(pinch.z*Math.hypot(a.x-b.x,a.y-b.y)/pinch.d,(a.x+b.x)/2-r.left,(a.y+b.y)/2-r.top);return}
 if(mode=='div'){d=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));draw()}
 else if(mode=='pan'){tx+=e.clientX-last.x;ty+=e.clientY-last.y;last={x:e.clientX,y:e.clientY};clamp();draw()}};
S.onpointerup=S.onpointercancel=e=>{pts.delete(e.pointerId);if(pts.size<2)pinch=null;mode=null;S.classList.remove('panning')};
S.ondblclick=e=>{if(e.target.id=='handle')return;const r=S.getBoundingClientRect();zoomAt(z>1.5?1:3,e.clientX-r.left,e.clientY-r.top)};
$('fit').onclick=fit;
$('z100').onclick=()=>{const r=S.getBoundingClientRect();zoomAt(fullW/(fw*devicePixelRatio),r.width/2,r.height/2)};
onresize=()=>fit();
onkeydown=e=>{if(['INPUT','SELECT','BUTTON'].includes(e.target.tagName))return;
 if(e.key=='ArrowRight')show(i+1);else if(e.key=='ArrowLeft')show(i-1);else if(e.key=='0')fit();
 else if(e.key==' '){e.preventDefault();d=d>.5?0:1;draw()}};

$('reset').onclick=()=>{p={...defs};sync();dirty=true;schedule()};
$('copy').onclick=()=>{clip={...p};msg('Ajustes copiados.')};
$('paste').onclick=()=>{if(!clip)return msg('Primero copiá ajustes de otra foto.');p={...clip};sync();dirty=true;schedule()};
$('save').onclick=async()=>{const ph=photos[i];msg('Guardando a resolución completa…');$('save').disabled=true;
 try{const r=await fetch('/api/save',{method:'POST',body:JSON.stringify({name:ph.name,params:p})});const j=await r.json();
  if(!r.ok)throw new Error(j.error);ph.params=j.params;ph.edited=true;dirty=false;ver++;marks();loadAfter();msg('Guardada en la carpeta de editadas.')}
 catch(x){msg('Error al guardar: '+x.message)}finally{$('save').disabled=false}};
</script></body></html>"""


# ---------- mejora con IA (ai_enhance.py) ----------

def _ai():
    import ai_enhance
    return ai_enhance


@_tool
def ai_retouch(path: str, strength: float = 1.0, output_path: str | None = None):
    """Retoque automático de exposición y color aprendido de fotógrafos
    profesionales (3D LUT adaptativa, MIT-Adobe FiveK). Rápido y natural.

    strength: 0.0 a 1.5 (1.0 = retoque completo del modelo).
    """
    img = _ai().lut_retouch(_load(path), strength)
    out = _out_path(path, "retouch", output_path)
    _save(img, out)
    return _result(img, out)


@_tool
def ai_restore(path: str, instruction: str, output_path: str | None = None):
    """Restaura la foto según una instrucción en INGLÉS (InstructIR).

    Sirve para: ruido ("Remove the noise"), desenfoque ("Fix the blur"),
    poca luz ("The photo is too dark, improve the illumination"), neblina
    ("Remove the haze"), lluvia ("Remove the rain") o mejora general
    ("Make the colors and contrast look professional"). Un problema por
    llamada; para varios, encadená llamadas (primero luz, después ruido).
    """
    img, understood = _ai().instructir(_load(path), instruction)
    out = _out_path(path, "restored", output_path)
    _save(img, out)
    return _result(img, out, f"El modelo lo interpretó como: {understood}.")


@_tool
def colorize(path: str, output_path: str | None = None):
    """Colorea una foto en blanco y negro (DDColor), conservando su detalle original."""
    img = _ai().colorize(_load(path))
    out = _out_path(path, "color", output_path)
    _save(img, out)
    return _result(img, out)


@_tool
def list_scenes():
    """Lista los perfiles de escena disponibles para enhance_scene y save_preset."""
    return "\n".join(f"{k}: {v['label']}" for k, v in _scenes().PROFILES.items())


@_tool
def detect_scene(path: str):
    """Detecta el tipo de escena (noche, nublado, atardecer, interior, contraluz,
    niebla, retrato, sol fuerte) con CLIP más datos EXIF y del histograma."""
    sc = _scenes()
    scene, probs = _scene_for(path)
    top = ", ".join(f"{k} {v * 100:.0f}%" for k, v in list(probs.items())[:4])
    return f"Escena: {scene} ({sc.PROFILES[scene]['label']}). Probabilidades: {top}."


@_tool
def enhance_scene(path: str, scene: str = "auto", amount: float = 1.0, output_path: str | None = None):
    """Mejora natural adaptada a la escena: cada perfil ajusta balance de blancos,
    exposición, sombras/luces sin halos, contraste, neblina, vibración (protege
    piel), ruido y nitidez de bordes con la intensidad adecuada para esa luz.

    scene: "auto" (detecta) o uno de list_scenes (nublado, noche, atardecer, …).
    amount: intensidad 0..1.5 (1 = la del perfil).
    """
    sc = _scenes()
    if not _check_scene(scene):
        raise ValueError("Indicá una escena: auto o una de list_scenes")
    img = _load(path)
    name, probs = _scene_for(path) if scene == "auto" else (scene, {})
    img = sc.enhance(img, name, amount)
    out = _out_path(path, f"scene_{name}", output_path)
    _save(img, out)
    note = f"Perfil: {sc.PROFILES[name]['label']}"
    if probs:
        note += f" (detectado, {probs.get(name, 0) * 100:.0f}% de confianza)"
    return _result(img, out, note + ".")


@_tool
def score_images(paths: list[str]):
    """Puntúa fotos con IA: calidad técnica (TOPIQ, 0-1: nitidez, ruido,
    exposición), estética (NIMA, 1-10) y un total 0-100. Útil para comparar
    versiones de una misma foto o elegir las mejores de un grupo."""
    ai = _ai()
    rows = [(p, ai.score(_load(p))) for p in paths]
    rows.sort(key=lambda r: -r[1]["total"])
    return "\n".join(f"{s['total']:5.1f}  calidad {s['calidad']:.3f}  estética {s['estetica']:.2f}  {p}"
                     for p, s in rows)


def _noise_sigma(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))))


_IIR = {"enhance": "Make the colors and contrast look professional",
        "light": "The photo is too dark, improve the illumination",
        "noise": "Remove the noise", "blur": "Fix the blur, make it sharper"}


def _recipes(img: np.ndarray, colorize_bw: bool, path) -> dict:
    """Combinaciones a probar según lo que se detecta en la foto."""
    ai = _ai()
    r = {"original": [], "auto 0.5": [("auto", 0.5)], "auto 1.0": [("auto", 1.0)],
         "LUT 0.7": [("lut", 0.7)], "LUT 1.0": [("lut", 1.0)],
         "LUT + auto 0.4": [("lut", 1.0), ("auto", 0.4)], "IIR mejora": [("iir", "enhance")]}
    gray = ai.looks_grayscale(img)
    dark = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean() < 80
    noisy = _noise_sigma(img) > 3.5
    scene, _ = _scene_for(path)
    r[f"escena {scene}"] = [("scene", (scene, 1.0))]
    r[f"escena {scene} 0.6"] = [("scene", (scene, 0.6))]
    if gray and colorize_bw:
        # las métricas no premian el color, así que solo compiten versiones coloreadas
        return ({"color": [("color", None)], "color + LUT 0.6": [("color", None), ("lut", 0.6)],
                 "color + auto 0.5": [("color", None), ("auto", 0.5)]},
                {"blanco y negro": True, "oscura": dark, "ruido": noisy})
    if dark:
        r["IIR luz"] = [("iir", "light")]
        r["IIR luz + ruido"] = [("iir", "light"), ("iir", "noise")]
        r["IIR luz + LUT 0.6"] = [("iir", "light"), ("lut", 0.6)]
    if noisy:
        r["IIR ruido"] = [("iir", "noise")]
        r["IIR ruido + LUT"] = [("iir", "noise"), ("lut", 1.0)]
    return r, {"blanco y negro": gray, "oscura": dark, "ruido": noisy}


def _run(img: np.ndarray, steps: list, scale: float = 1.0) -> np.ndarray:
    ai = _ai()
    for op, arg in steps:
        if op == "auto":
            img = _enhance(img, arg, scale)
        elif op == "lut":
            img = ai.lut_retouch(img, arg)
        elif op == "iir":
            img = ai.instructir(img, _IIR[arg])[0]
        elif op == "color":
            img = ai.colorize(img)
        elif op == "scene":
            img = _scenes().enhance(img, arg[0], arg[1], scale)
    return img


@_tool
def best_enhance(path: str, extra_instructions: list[str] | None = None, colorize_bw: bool = False,
                 output_path: str | None = None):
    """Mejora automática inteligente: detecta problemas (oscura, ruido, blanco y
    negro), prueba varias combinaciones de retoque (LUT, InstructIR, mejora
    clásica), las puntúa con IA y aplica la mejor a resolución completa.
    Si ninguna mejora claramente al original (+1.5 puntos), la deja como está.

    extra_instructions: instrucciones en inglés para InstructIR a probar además
    (por ejemplo "Remove the haze").
    colorize_bw: si la foto es blanco y negro, colorearla y elegir la mejor
    versión coloreada.
    """
    full = _load(path)
    h, w = full.shape[:2]
    k = min(1.0, 1024 / max(h, w))
    small = cv2.resize(full, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else full
    recipes, found = _recipes(small, colorize_bw, path)
    for i, ins in enumerate(extra_instructions or []):
        _IIR[f"extra{i}"] = ins
        recipes[f"IIR: {ins[:30]}"] = [("iir", f"extra{i}")]
    ai = _ai()
    results = {}
    for name, steps in recipes.items():
        cand = _run(small, steps, k)
        results[name] = (ai.score(cand), cand)
    ranking = sorted(results.items(), key=lambda kv: -kv[1][0]["total"])
    best = ranking[0][0]
    if "original" in results and ranking[0][1][0]["total"] < results["original"][0]["total"] + 1.5:
        best = "original"
    img = _run(full, recipes[best]) if recipes[best] else full
    out = _out_path(path, "best", output_path)
    _save(img, out)
    # hoja con las 4 mejores para que se vea la elección
    top = [_label(c.copy(), f"{n}: {s['total']:.1f}") for n, (s, c) in ranking[:4]]
    th = min(t.shape[0] for t in top)
    sheet = np.concatenate([cv2.resize(t, (round(t.shape[1] * th / t.shape[0]), th)) for t in top], axis=1)
    detected = ", ".join(k for k, v in found.items() if v) or "nada en particular"
    table = "\n".join(f"  {s['total']:5.1f}  {n}" for n, (s, _) in ranking)
    vs = f" vs original {results['original'][0]['total']:.1f}" if "original" in results else ""
    msg = (f"Detectado: {detected}. Elegida: «{best}» ({results[best][0]['total']:.1f}{vs}). "
           f"Guardado en {out} ({img.shape[1]}x{img.shape[0]}).\nPuntajes (0-100):\n{table}")
    if best == "original" and len(results) > 1:
        msg += "\nNinguna versión mejoró claramente la foto; se guardó sin cambios."
    if found["blanco y negro"] and not colorize_bw:
        msg += "\nEs blanco y negro: usá colorize_bw=true (o la herramienta colorize) para colorearla."
    return [msg, _preview(sheet, 1600)]


if __name__ == "__main__":
    # python server.py viewer <originales> <editadas> [puerto]: visor sin el MCP
    if len(sys.argv) >= 4 and sys.argv[1] == "viewer":
        print(open_viewer(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 8765), flush=True)
        threading.Event().wait()
    else:
        mcp.run()
