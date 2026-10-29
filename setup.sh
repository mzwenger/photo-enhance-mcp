#!/usr/bin/env bash
# Instala el entorno de Python y descarga los modelos (~2 GB) en esta carpeta.
# Uso: ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null; then
  echo "Instalando uv (gestor de Python de Astral) en ~/.local/bin…"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt

get() {  # get <url> <destino>
  [ -s "$2" ] && return 0
  echo "Descargando $(basename "$2")…"
  mkdir -p "$(dirname "$2")"
  curl -sSLf -o "$2.part" "$1" && mv "$2.part" "$2"
}

HF=https://huggingface.co
GH=https://github.com
get "$HF/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"                  models/lama_fp32.onnx
get "$HF/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx"         models/gfpgan_1.4.onnx
get "$HF/facefusion/models-3.0.0/resolve/main/ddcolor.onnx"            models/ddcolor.onnx
get "$GH/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    models/face_detection_yunet_2023mar.onnx
get "$GH/mv-lab/InstructIR/raw/main/models/im_instructir-7d.pt"        models/im_instructir-7d.pt
get "$GH/mv-lab/InstructIR/raw/main/models/lm_instructir-7d.pt"        models/lm_instructir-7d.pt
get "$GH/HuiZeng/Image-Adaptive-3DLUT/raw/master/pretrained_models/sRGB/LUTs.pth"       models/lut/LUTs.pth
get "$GH/HuiZeng/Image-Adaptive-3DLUT/raw/master/pretrained_models/sRGB/classifier.pth" models/lut/classifier.pth

if [ ! -x bin/realesrgan/realesrgan-ncnn-vulkan ]; then
  get "$GH/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip" bin/realesrgan.zip
  mkdir -p bin/realesrgan && (cd bin/realesrgan && unzip -oq ../realesrgan.zip) && rm bin/realesrgan.zip
  chmod +x bin/realesrgan/realesrgan-ncnn-vulkan
fi

# rembg descarga su modelo la primera vez; lo dejamos listo
U2NET_HOME="$PWD/models/rembg" .venv/bin/python -c "from rembg import new_session; new_session('isnet-general-use')"

echo
echo "Listo. Para registrar el servidor MCP en Claude Code:"
echo "  claude mcp add --scope user photo-enhance -- $PWD/.venv/bin/python $PWD/server.py"
