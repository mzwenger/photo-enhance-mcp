# photo-enhance-mcp

[![Prueba multiplataforma](https://github.com/mzwenger/photo-enhance-mcp/actions/workflows/multiplataforma.yml/badge.svg)](https://github.com/mzwenger/photo-enhance-mcp/actions/workflows/multiplataforma.yml)

Servidor [MCP](https://modelcontextprotocol.io) local para editar y mejorar fotos con IA, más un visor web
antes/después para ajustar y procesar carpetas enteras. Todo corre en tu computadora (CPU; GPU Intel/NVIDIA
opcional vía OpenVINO/Vulkan), sin enviar fotos a ningún servicio.

*Local MCP server for AI photo editing and enhancement, plus a before/after web viewer. Runs fully offline.*

## Qué hace

**Mejora y restauración**

| Herramienta | Qué hace | Modelo |
|---|---|---|
| `enhance_scene` | Mejora natural según la escena (nublado, noche, atardecer, interior, contraluz, niebla, retrato, sol fuerte) o detección automática | Filtro guiado + CLIP |
| `detect_scene`, `list_scenes` | Detecta el tipo de escena (CLIP + EXIF + histograma) | CLIP ViT-B/32 |
| `save_scene_profile`, `delete_scene_profile` | Perfiles de escena propios con parámetros finos (luces, sombras, calidez, bruma…), guardados en `profiles.json` | — |
| `auto_tune` | Busca los parámetros óptimos de una foto con un procedimiento determinista (calidad, detalle por zona, recortes, pérdida de color, tinte, grano al 100%) y opcionalmente la guarda; si nada la mejora, la deja igual | autotune.py + pyiqa |
| `auto_tune_batch`, `tune_status`, `tune_cancel` | Lo mismo para una carpeta o lista de fotos, en segundo plano: cada foto con sus parámetros (`per_photo`) o un ajuste común para el conjunto (`group`). Los originales nunca se tocan | autotune.py |
| `best_enhance` | Prueba varias combinaciones, las puntúa con IA y aplica la mejor | todas + pyiqa |
| `ai_retouch` | Retoque de color y exposición aprendido de fotógrafos (MIT-Adobe FiveK) | Image-Adaptive 3D LUT |
| `ai_restore` | Restauración por instrucción en inglés: ruido, desenfoque, poca luz, neblina, lluvia | InstructIR |
| `colorize` | Colorea fotos en blanco y negro | DDColor |
| `upscale` | Aumenta la resolución ×2/×3/×4 | Real-ESRGAN (Vulkan) |
| `restore_faces` | Restaura caras borrosas o de fotos viejas | GFPGAN 1.4 + YuNet |
| `remove_objects` | Borra objetos y rellena el hueco | LaMa |
| `remove_background` | Quita o reemplaza el fondo | rembg (ISNet) |
| `score_images` | Puntaje de calidad técnica y estética (0-100) | TOPIQ + NIMA (pyiqa) |

**Edición clásica:** `auto_enhance`, `adjust` (brillo, contraste, saturación, calidez, gamma, nitidez),
`denoise`, `transform` (recortar, redimensionar, rotar, convertir), `image_info`.

**Comparar y procesar en lote:** `compare`, `compare_gallery`, `open_viewer`, `save_preset`,
`list_presets`, `apply_preset`.

Cada herramienta devuelve una vista previa para que el agente vea el resultado. Las fotos originales nunca
se modifican: todo lo que se genera va, por defecto, a la subcarpeta **`mejoradas/`** de la carpeta de las
fotos (se crea sola). Las herramientas que procesan carpetas y el visor guardan con el mismo nombre que el
original; las herramientas de una sola foto agregan un sufijo (`DSC_0056_scene_nublado.jpg`) para no pisar
otras versiones. Todas aceptan otra carpeta o ruta de salida si se indica.

## Visor web

- Deslizador antes/después con zoom y paneo.
- Histograma RGB y de luminosidad con recorte de sombras, luces y color medido por canal, y comparación con la original.
- Cada control muestra en su barra: verde sin recorte, amarillo recorte tolerable (manchas chicas y dispersas, como reflejos), rojo recorte visible (zonas continuas, como un cielo quemado) y gris recorte visible causado por otros ajustes. Los umbrales están en `clipping.py`.
- Recorte y rotación: rectángulo con proporciones fijas (libre, original, 1:1, 4:3, 3:2, 16:9, 4:5, 2:3), enderezado en tiempo real (±45°, sin esquinas vacías) y giros de 90°. Si el recorte deja la foto chica, al guardar se reescala con IA (Real-ESRGAN) hasta cerca del tamaño original. El recorte es de cada foto: no se copia en presets ni en lotes, y los lotes conservan el recorte que ya tenía cada foto.
- Perfiles de escena (automático o manual) con intensidad regulable.
- Controles en vivo: mejora automática, brillo, contraste, saturación, calidez, gamma, nitidez.
- Ajustes guardados con nombre y aplicación en lote a las fotos seleccionadas.
- Siempre parte del original: los ajustes no se acumulan.

```bash
.venv/bin/python server.py viewer ~/Fotos/viaje            # editadas en ~/Fotos/viaje/mejoradas
.venv/bin/python server.py viewer ~/Fotos/viaje ~/otra 8770  # otra carpeta de salida y otro puerto
```

Abre `http://127.0.0.1:8765/`. También se puede abrir desde el agente con la herramienta `open_viewer`
(en ese caso vive mientras viva el servidor MCP).

En Windows: `.venv\Scripts\python.exe server.py viewer C:\Fotos\viaje`.

Para que siga corriendo aunque cierres la terminal o el agente (hasta reiniciar la PC), en Linux:

```bash
systemd-run --user --unit=visor-fotos "$PWD/.venv/bin/python" "$PWD/server.py" viewer ~/Fotos/viaje
systemctl --user stop visor-fotos   # para apagarlo
```

En Windows, sin ventana de consola: `Start-Process .venv\Scripts\pythonw.exe -ArgumentList 'server.py','viewer','C:\Fotos\viaje'`.

## Instalación

Funciona en **Linux, Windows y macOS** (x86_64). Cada cambio se prueba automáticamente en Linux y
Windows con GitHub Actions, instalando desde una carpeta con tildes y eñes en el nombre.
Memoria: 8 GB de RAM como mínimo. Un perfil de escena sobre una foto de 24 MP usa ~4 GB, por eso el
procesamiento en lote con perfil de escena trabaja de a una foto (sin perfil, de a tres).

**Linux / macOS**

```bash
git clone https://github.com/mzwenger/photo-enhance-mcp
cd photo-enhance-mcp
./setup.sh
```

**Windows** (PowerShell)

```powershell
git clone https://github.com/mzwenger/photo-enhance-mcp
cd photo-enhance-mcp
powershell -ExecutionPolicy Bypass -File setup.ps1
```

El instalador (`install.py`) instala [uv](https://docs.astral.sh/uv/) si falta, crea un entorno Python 3.12
(no hace falta tener Python instalado), instala PyTorch para CPU y descarga los modelos (~2 GB) en
`models/` y `bin/`, incluido Real-ESRGAN para tu sistema. Algunos modelos (CLIP, métricas de pyiqa, el
codificador de texto de InstructIR) se descargan solos la primera vez que se usan.

Real-ESRGAN usa la GPU por Vulkan (NVIDIA, AMD e Intel tienen drivers con Vulkan). Si no hay Vulkan
disponible, `upscale` usa un reescalado clásico (Lanczos) y lo avisa en vez de fallar.

### Conectarlo a Claude Code

Linux / macOS:

```bash
claude mcp add --scope user photo-enhance -- "$PWD/.venv/bin/python" "$PWD/server.py"
```

Windows:

```powershell
claude mcp add --scope user photo-enhance -- "$PWD\.venv\Scripts\python.exe" "$PWD\server.py"
```

O agregá a `~/.claude.json` (en Windows, `%USERPROFILE%\.claude.json`, con rutas como
`C:\\Users\\vos\\photo-enhance-mcp\\.venv\\Scripts\\python.exe`):

```json
{
  "mcpServers": {
    "photo-enhance": {
      "type": "stdio",
      "command": "/ruta/a/photo-enhance-mcp/.venv/bin/python",
      "args": ["/ruta/a/photo-enhance-mcp/server.py"]
    }
  }
}
```

Después pedile al agente cosas como *"mejorá las fotos de ~/Fotos/viaje con el perfil automático"* o
*"borrá el cartel de esta foto y subile la resolución"*.

### Aceleración por hardware (opcional)

- **GPU Intel:** `sudo apt install intel-opencl-icd libze-intel-gpu1` y OpenVINO la usa sola.
- Para forzar un dispositivo: `IMAGE_KIT_DEVICE=CPU|GPU|NPU`.
- Memoria del servidor: los modelos quedan cargados para responder rápido y se descargan solos
  cuando el proceso supera `IMAGE_KIT_MAX_RAM_GB` (3 por defecto). Con más RAM, subirlo acelera
  las llamadas seguidas.

## Estructura

```
server.py       servidor MCP, herramientas clásicas y visor web
scenes.py       mejora natural por escena y detección de escena
autotune.py     búsqueda determinista de parámetros óptimos
clipping.py     recorte tolerable vs visible (tamaño de mancha y total por tipo)
ai_enhance.py   3D LUT, InstructIR, DDColor y puntaje de calidad
ai/instructir/  arquitectura de InstructIR (MIT, ver su LICENSE)
install.py      instalación multiplataforma y descarga de modelos
setup.sh        instalador para Linux y macOS (instala uv y ejecuta install.py)
setup.ps1       instalador para Windows
tests/          prueba de humo (MCP y visor), la misma que corre GitHub Actions
```

## Licencias

El código de este repositorio es MIT (ver [LICENSE](LICENSE)). Los modelos y algunas dependencias tienen
licencias propias que se aplican a su uso:

| Componente | Licencia |
|---|---|
| InstructIR, CLIP, rembg, YuNet | MIT / Apache-2.0 |
| LaMa, GFPGAN, DDColor, Image-Adaptive 3D LUT, OpenVINO | Apache-2.0 |
| Real-ESRGAN | BSD-3-Clause |
| **pyiqa** (usado por `score_images` y `best_enhance`) | **PolyForm Noncommercial 1.0.0**: solo uso no comercial |

Revisá la licencia de cada proyecto antes de un uso comercial.

## Créditos

[LaMa](https://github.com/advimman/lama) ·
[GFPGAN](https://github.com/TencentARC/GFPGAN) ·
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) ·
[rembg](https://github.com/danielgatis/rembg) ·
[InstructIR](https://github.com/mv-lab/InstructIR) ·
[DDColor](https://github.com/piddnad/DDColor) ·
[Image-Adaptive 3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) ·
[IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) ·
[CLIP](https://github.com/openai/CLIP) ·
[OpenCV Zoo](https://github.com/opencv/opencv_zoo) ·
exportes ONNX de [FaceFusion](https://huggingface.co/facefusion) y [Carve](https://huggingface.co/Carve/LaMa-ONNX).
