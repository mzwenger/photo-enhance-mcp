"""Prueba de humo multiplataforma: herramientas MCP principales y visor web.

Usa una carpeta con tildes y eñes en el nombre (en Windows OpenCV no abre esas rutas si se
le pasan directo) y verifica que los originales no cambien y que todo vaya a mejoradas/.
Se ejecuta con el Python del entorno:  .venv/bin/python tests/smoke_test.py
(en Windows: .venv\\Scripts\\python.exe tests\\smoke_test.py)
"""

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

KIT = Path(__file__).resolve().parents[1]
PY = sys.executable
SAMPLES = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
failures: list[str] = []


def log(msg):
    print(msg, flush=True)


def check(cond, msg):
    log(("OK   " if cond else "FALLA") + " " + msg)
    if not cond:
        failures.append(msg)


def write_jpg(path: Path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    buf.tofile(str(path))  # tofile/fromfile: rutas con tildes también en Windows


def fetch(name):
    with urllib.request.urlopen(SAMPLES + name, timeout=60) as r:
        return cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)


def digest(folder: Path):
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.iterdir() if f.is_file()}


async def mcp_tests(fotos: Path):
    params = StdioServerParameters(command=PY, args=[str(KIT / "server.py")],
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        names = {t.name for t in (await s.list_tools()).tools}
        check(len(names) >= 29, f"el MCP expone {len(names)} herramientas")

        async def call(name, args, expect_ok=True):
            t0 = time.time()
            res = await s.call_tool(name, args)
            txt = " ".join(c.text for c in res.content if c.type == "text").replace("\n", " | ")
            check(res.is_error != expect_ok, f"{name} ({time.time() - t0:.0f} s): {txt[:150]}")
            return txt

        f = lambda n: str(fotos / n)
        await call("image_info", {"path": f("fútbol.jpg")})
        await call("adjust", {"path": f("fútbol.jpg"), "brightness": 10, "sharpness": 30})
        await call("auto_enhance", {"path": f("fútbol.jpg")})
        await call("enhance_scene", {"path": f("fútbol.jpg"), "scene": "nublado"})
        await call("detect_scene", {"path": f("fútbol.jpg")})
        await call("remove_objects", {"path": f("fútbol.jpg"), "boxes": [[334, 288, 56, 52]]})
        await call("remove_background", {"path": f("fútbol.jpg")})
        await call("restore_faces", {"path": f("retrato año.jpg")})
        await call("colorize", {"path": f("blanco y negro.jpg")})
        await call("upscale", {"path": f("chica.jpg"), "scale": 2})
        await call("ai_retouch", {"path": f("fútbol.jpg")})
        await call("ai_restore", {"path": f("chica.jpg"), "instruction": "Remove the noise"})
        await call("score_images", {"paths": [f("fútbol.jpg"), f("chica.jpg")]})
        await call("save_preset", {"name": "_smoke", "scene": "auto"})
        await call("apply_preset", {"original_dir": str(fotos), "preset": "_smoke", "files": ["fútbol.jpg"]})
        await call("list_presets", {})
        await call("auto_tune", {"path": f("chica.jpg"), "apply": True})
        await call("auto_tune_batch", {"original_dir": str(fotos), "files": ["fútbol.jpg"], "mode": "group"})
        for _ in range(240):
            txt = (await s.call_tool("tune_status", {})).content[0].text
            if "en curso" not in txt.splitlines()[0]:
                break
            await asyncio.sleep(5)
        check("terminada" in txt.splitlines()[0], "auto_tune_batch terminó: " + txt.splitlines()[0])
        await call("apply_preset", {"original_dir": str(fotos), "preset": "_smoke", "edited_dir": str(fotos)},
                   expect_ok=False)  # nunca escribir sobre los originales


def free_port():
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read(), r.headers


def viewer_tests(fotos: Path):
    port = free_port()
    proc = subprocess.Popen([PY, str(KIT / "server.py"), "viewer", str(fotos), str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                get(base + "/api/presets", 2)
                break
            except OSError:
                time.sleep(.5)
        st, body, hd = get(base + "/")
        check(st == 200 and b"Visor" in body and hd.get("X-Viewer"), "visor: página")
        lst = json.loads(get(base + "/api/list")[1])
        check(len(lst["photos"]) == 4 and lst["dst"].endswith("mejoradas"), "visor: lista de fotos y carpeta mejoradas")
        q = "name=" + urllib.request.quote("fútbol.jpg")
        st, body, _ = get(base + f"/preview?{q}&w=800&scene=nublado")
        check(st == 200 and body[:2] == b"\xff\xd8", "visor: vista previa con perfil")
        safe = json.loads(get(base + f"/api/safe?{q}&cid=smoke&brightness=20")[1])
        check("brightness" in safe and "_total" in safe, "visor: rangos sin pérdida")
        sc = json.loads(get(base + f"/api/scene?{q}")[1])
        check("scene" in sc, f"visor: escena detectada ({sc.get('scene')})")
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    log(f"Python {sys.version.split()[0]} en {sys.platform}; kit en {KIT}")
    with tempfile.TemporaryDirectory() as td:
        fotos = Path(td) / "fotos ñandú áé"
        fotos.mkdir()
        messi = fetch("messi5.jpg")
        write_jpg(fotos / "fútbol.jpg", messi)
        write_jpg(fotos / "retrato año.jpg", fetch("lena.jpg"))
        write_jpg(fotos / "blanco y negro.jpg", cv2.cvtColor(cv2.cvtColor(messi, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
        small = cv2.resize(messi, (274, 171), interpolation=cv2.INTER_AREA)
        noisy = np.clip(small.astype(np.float32) + np.random.default_rng(0).normal(0, 8, small.shape), 0, 255)
        write_jpg(fotos / "chica.jpg", noisy.astype(np.uint8))
        before = digest(fotos)
        try:
            asyncio.run(mcp_tests(fotos))
            viewer_tests(fotos)
        finally:
            sys.path.insert(0, str(KIT))
            import server  # quitar el preset de prueba (presets.json es del usuario)
            try:
                server._store_preset("_smoke", None)
            except Exception:
                pass
        check(digest(fotos) == before, "los originales no cambiaron y no se agregaron archivos junto a ellos")
        made = sorted(p.name for p in (fotos / "mejoradas").iterdir())
        check(len(made) >= 10, f"salidas en mejoradas/: {len(made)} archivos")
    log(f"\n{'TODO OK' if not failures else f'{len(failures)} FALLAS'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
