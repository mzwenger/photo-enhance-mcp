"""Instalación multiplataforma (Linux, Windows, macOS): entorno de Python y modelos (~2 GB).

Se ejecuta con el Python que provee uv (ver setup.sh / setup.ps1):
    uv run --no-project --python 3.12 install.py [--no-models]
Solo usa la biblioteca estándar.
"""

import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parent
WIN = os.name == "nt"
MAC = sys.platform == "darwin"
VENV = KIT / ".venv"
PY = VENV / ("Scripts/python.exe" if WIN else "bin/python")

HF = "https://huggingface.co"
GH = "https://github.com"
MODELS = [
    (f"{HF}/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx", "models/lama_fp32.onnx"),
    (f"{HF}/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx", "models/gfpgan_1.4.onnx"),
    (f"{HF}/facefusion/models-3.0.0/resolve/main/ddcolor.onnx", "models/ddcolor.onnx"),
    (f"{GH}/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
     "models/face_detection_yunet_2023mar.onnx"),
    (f"{GH}/mv-lab/InstructIR/raw/main/models/im_instructir-7d.pt", "models/im_instructir-7d.pt"),
    (f"{GH}/mv-lab/InstructIR/raw/main/models/lm_instructir-7d.pt", "models/lm_instructir-7d.pt"),
    (f"{GH}/HuiZeng/Image-Adaptive-3DLUT/raw/master/pretrained_models/sRGB/LUTs.pth", "models/lut/LUTs.pth"),
    (f"{GH}/HuiZeng/Image-Adaptive-3DLUT/raw/master/pretrained_models/sRGB/classifier.pth",
     "models/lut/classifier.pth"),
    (f"{GH}/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
     "models/esrgan/realesr-general-x4v3.pth"),
    (f"{GH}/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
     "models/esrgan/realesr-general-wdn-x4v3.pth"),
]
ESRGAN = (f"{GH}/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-"
          + ("windows" if WIN else "macos" if MAC else "ubuntu") + ".zip")


def say(msg: str) -> None:
    print(msg, flush=True)


def run(*cmd, env=None) -> None:
    say("> " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=KIT, env=env)


def download(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    say(f"Descargando {dest.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "photo-enhance-mcp-install"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total, done, step = int(r.headers.get("Content-Length") or 0), 0, 10
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and done * 100 // total >= step:
                say(f"  {dest.name}: {done * 100 // total}%")
                step += 10
    os.replace(part, dest)


def main() -> None:
    try:  # la consola de Windows puede no ser UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    uv = shutil.which("uv")
    if not uv:
        sys.exit("No se encontró uv. Usá setup.sh (Linux/macOS) o setup.ps1 (Windows), que lo instalan.")
    say(f"Sistema: {platform.system()} {platform.machine()} | carpeta: {KIT}")

    if not PY.is_file():
        run(uv, "venv", "--python", "3.12", VENV)
    torch_src = [] if MAC else ["--index-url", "https://download.pytorch.org/whl/cpu"]
    run(uv, "pip", "install", "--python", PY, "torch", "torchvision", *torch_src)
    run(uv, "pip", "install", "--python", PY, "-r", KIT / "requirements.txt")

    if "--no-models" in sys.argv:
        say("Modelos omitidos (--no-models).")
    else:
        for url, rel in MODELS:
            download(url, KIT / rel)
        exe = KIT / "bin" / "realesrgan" / ("realesrgan-ncnn-vulkan.exe" if WIN else "realesrgan-ncnn-vulkan")
        if not exe.is_file():
            z = KIT / "bin" / "realesrgan.zip"
            download(ESRGAN, z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(KIT / "bin" / "realesrgan")
            z.unlink()
            if not WIN:
                exe.chmod(0o755)
        # rembg descarga su modelo la primera vez; lo dejamos listo
        env = {**os.environ, "U2NET_HOME": str(KIT / "models" / "rembg")}
        run(PY, "-c", "from rembg import new_session; new_session('isnet-general-use')", env=env)

    q = (lambda s: f'"{s}"')
    say("\nListo. Para registrar el servidor MCP en Claude Code:")
    say(f"  claude mcp add --scope user photo-enhance -- {q(PY)} {q(KIT / 'server.py')}")
    say("Visor web:")
    say(f"  {q(PY)} {q(KIT / 'server.py')} viewer <carpeta de fotos>")


if __name__ == "__main__":
    main()
