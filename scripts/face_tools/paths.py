import re

def normalize_texture(path):
    path = path.replace('\\', '/').lower()
    path = re.sub(r'^/assets?/(?:pes\d+/)?', '/asset/', path)
    path = path.replace('/#windx11/', '/')
    return path

