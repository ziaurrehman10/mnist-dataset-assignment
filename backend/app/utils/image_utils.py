# api/app/utils/image_utils.py  ← FIXED VERSION
import io
import numpy as np
from PIL import Image, ImageOps

def image_bytes_to_array(raw_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw_bytes)).convert("L")  # greyscale

    # ── KEY FIX: check background color and invert if needed ──────────────────
    # MNIST = white digit on BLACK background
    # Most uploads/drawings = black digit on WHITE background → must invert
    corners = [
        img.getpixel((0, 0)),
        img.getpixel((img.width-1, 0)),
        img.getpixel((0, img.height-1)),
        img.getpixel((img.width-1, img.height-1)),
    ]
    avg_corner = sum(corners) / 4
    if avg_corner > 128:           # white background → invert
        img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten()