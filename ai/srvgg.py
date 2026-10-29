"""Real-ESRGAN realesr-general-x4v3 (Wang et al., BSD-3): red compacta SRVGGNet para fotos reales.

Más liviana que RRDBNet (x4plus) y con menos aspecto "plástico" en texturas finas. Se mezclan
los pesos normales y los de reducción de ruido (wdn) según `denoise` (0 = conserva todo el
grano/detalle, 1 = limpia más), como hace la herramienta oficial con su parámetro "dni".
"""

import torch
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def load(general_path, wdn_path, denoise: float) -> SRVGGNetCompact:
    a = torch.load(general_path, map_location="cpu", weights_only=True)["params"]
    b = torch.load(wdn_path, map_location="cpu", weights_only=True)["params"]
    s = float(min(max(denoise, 0.0), 1.0))
    net = SRVGGNetCompact()
    net.load_state_dict({k: (1 - s) * a[k] + s * b[k] for k in a})
    return net.eval()
