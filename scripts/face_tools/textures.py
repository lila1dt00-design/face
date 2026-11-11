import os,math
from PIL import Image

def limit_image(image, max_side=4096):
    """Downscale by longest side AND total pixel count (phones choke on huge atlases).
    Size is snapped down to powers of two when a resize is needed; UVs are normalised so this is safe."""
    if max_side < 1:
        raise ValueError('max_side must be positive')
    max_pixels = int(os.environ.get('FACE_MAX_PIXELS', 4194304))
    w, h = image.size
    factor = min(1.0, max_side / max(w, h), math.sqrt(max_pixels / (w * h)))
    if factor >= 1.0:
        return image
    def snap(v):
        v = max(1, int(v * factor)); return 1 << (v.bit_length() - 1)
    return image.resize((snap(w), snap(h)), Image.Resampling.LANCZOS)
