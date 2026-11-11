"""Distinguish opaque scalp skin from alpha-masked Hair2 shell overlays."""
from pathlib import Path
import re
from PIL import Image


def uses_alpha(material, base_name, image):
    if base_name in ['fox_hair_mat', 'fox_hair_transfer_mat', 'fox_hair_decal_mat']:
        return True
    if base_name != 'fox_head_shell_mat':
        return False
    stem=re.sub(r'^\d+_', '', Path(image).stem.lower())
    if stem.startswith('hair'):
        # Dedicated hair textures may be entirely transparent to hide a donor mesh.
        return True
    shader = (getattr(material, 'shader', '') + ' ' + getattr(material, 'technique', '')).lower()
    if 'hair' not in shader:
        return False
    with Image.open(image) as original:
        low, high = original.convert('RGBA').getchannel('A').getextrema()
    # A zero-alpha face map is common opaque skin, while a varying alpha on
    # a Hair2 shell describes beard, eyebrow or scalp overlay opacity.
    return low < high
