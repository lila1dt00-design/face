from pathlib import Path
import json,struct
import numpy as np
from PIL import Image
DTYPES={5120:"i1",5121:"u1",5122:"<i2",5123:"<u2",5125:"<u4",5126:"<f4"}
COMPONENTS={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT4":16}

def read_glb(p):
    raw = p.read_bytes()
    assert struct.unpack_from('<4sII', raw) == (b'glTF', 2, len(raw))
    chunks = {}; at = 12
    while at < len(raw):
        size, kind = struct.unpack_from('<II', raw, at)
        chunks[kind] = raw[at + 8:at + 8 + size]; at += 8 + size
    return json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]


def accessor(g, data, ident):
    a = g['accessors'][ident]; v = g['bufferViews'][a['bufferView']]
    assert not a.get('sparse')
    dtype = np.dtype(DTYPES[a['componentType']]); width = COMPONENTS[a['type']]
    stride = v.get('byteStride', width * dtype.itemsize)
    result = np.ndarray((a['count'], width), dtype=dtype, buffer=data,
                        offset=v.get('byteOffset', 0) + a.get('byteOffset', 0), strides=(stride, dtype.itemsize)).copy()
    if a.get('normalized') and a['componentType'] != 5126:
        info = np.iinfo(dtype); result = result.astype(float) / info.max
        if info.min < 0: result = np.maximum(result, -1)
    return result


def prune_unused(g, data):
    """Drop accessors / bufferViews / binary bytes that nothing references (dead source data)."""
    data = bytes(data)
    used_acc = set()
    def acc(i):
        if i is not None: used_acc.add(i)
    for m in g.get('meshes', []):
        for pr in m.get('primitives', []):
            for i in pr.get('attributes', {}).values(): acc(i)
            acc(pr.get('indices'))
            for t in pr.get('targets', []):
                for i in t.values(): acc(i)
    for sk in g.get('skins', []): acc(sk.get('inverseBindMatrices'))
    for an in g.get('animations', []):
        for sm in an.get('samplers', []): acc(sm.get('input')); acc(sm.get('output'))
    if any(a.get('sparse') for a in g.get('accessors', [])): return g, data  # unusual file: leave untouched
    acc_map = {old: new for new, old in enumerate(sorted(used_acc))}
    used_views = {g['accessors'][i]['bufferView'] for i in used_acc if 'bufferView' in g['accessors'][i]}
    used_views |= {im['bufferView'] for im in g.get('images', []) if 'bufferView' in im}
    out = bytearray(); view_map = {}; views = []
    for old in sorted(used_views):
        v = dict(g['bufferViews'][old]); start = v.get('byteOffset', 0)
        out.extend(b'\0' * (-len(out) % 4)); v['byteOffset'] = len(out)
        out.extend(data[start:start + v['byteLength']]); view_map[old] = len(views); views.append(v)
    accessors = []
    for old in sorted(used_acc):
        a = dict(g['accessors'][old])
        if 'bufferView' in a: a['bufferView'] = view_map[a['bufferView']]
        accessors.append(a)
    for m in g.get('meshes', []):
        for pr in m.get('primitives', []):
            pr['attributes'] = {k: acc_map[i] for k, i in pr['attributes'].items()}
            if 'indices' in pr: pr['indices'] = acc_map[pr['indices']]
            if 'targets' in pr: pr['targets'] = [{k: acc_map[i] for k, i in t.items()} for t in pr['targets']]
    for sk in g.get('skins', []):
        if 'inverseBindMatrices' in sk: sk['inverseBindMatrices'] = acc_map[sk['inverseBindMatrices']]
    for an in g.get('animations', []):
        for sm in an.get('samplers', []): sm['input'] = acc_map[sm['input']]; sm['output'] = acc_map[sm['output']]
    for im in g.get('images', []):
        if 'bufferView' in im: im['bufferView'] = view_map[im['bufferView']]
    g['accessors'] = accessors; g['bufferViews'] = views
    g['buffers'] = [{'byteLength': len(out)}]
    return g, bytes(out)


def write_glb(p, g, data):
    g, data = prune_unused(g, data)
    js = json.dumps(g, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    js += b' ' * (-len(js) % 4); data += b'\0' * (-len(data) % 4)
    p.write_bytes(struct.pack('<4sII', b'glTF', 2, 28 + len(js) + len(data)) +
                  struct.pack('<II', len(js), 0x4E4F534A) + js +
                  struct.pack('<II', len(data), 0x004E4942) + data)


def local_matrix(node):
    if 'matrix' in node: return np.array(node['matrix'], dtype=float).reshape(4, 4).T
    x, y, z, w = node.get('rotation', [0, 0, 0, 1])
    rotation = np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                         [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                         [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]])
    m = np.eye(4); m[:3, :3] = rotation @ np.diag(node.get('scale', [1, 1, 1])); m[:3, 3] = node.get('translation', [0, 0, 0])
    return m


def world_matrices(g):
    result = {}
    def walk(i, parent):
        result[i] = parent @ local_matrix(g['nodes'][i])
        for child in g['nodes'][i].get('children', []): walk(child, result[i])
    for i in g['scenes'][g.get('scene', 0)]['nodes']: walk(i, np.eye(4))
    return result


def append_accessor(g, data, values, kind, component=5126, target=34962):
    values = np.asarray(values, dtype=DTYPES[component])
    data += b'\0' * (-len(data) % 4)
    view = len(g['bufferViews'])
    g['bufferViews'].append({'buffer': 0, 'byteOffset': len(data), 'byteLength': values.nbytes, 'target': target})
    data.extend(values.tobytes())
    ident = len(g['accessors'])
    g['accessors'].append({'bufferView': view, 'componentType': component, 'count': len(values), 'type': kind,
                           'min': values.min(axis=0).reshape(-1).tolist(), 'max': values.max(axis=0).reshape(-1).tolist()})
    return ident


def make_eye_texture(path):
    # Neutral dark-brown iris for offline viewing. This is a new texture, not
    # a claim to reproduce the game's dynamic eye shader or exact eye colour.
    y, x = np.mgrid[0:256, 0:256] / 255
    x = (x-.5)*2; y = (y-.5)*2; r = np.hypot(x, y); angle = np.arctan2(y, x)
    rgb = np.zeros((256, 256, 3)); rgb[:] = [.77, .75, .69]
    rgb -= np.minimum(r, 1)[..., None] * np.array([.09, .11, .12])
    iris = r < .52; pupil = r < .21
    fibres = np.sin(angle*91 + np.sin(angle*23)*2 + r*18)*.035 + np.sin(angle*157-r*15)*.018
    iris_rgb = np.stack([.20 + fibres, .115 + fibres*.7, .055 + fibres*.4], axis=-1)
    rim = np.clip((r-.45)/.07, 0, 1)
    iris_rgb *= (1-rim[..., None]*.62)
    rgb[iris] = iris_rgb[iris]; rgb[pupil] = [.012, .010, .009]
    rgb = np.clip(rgb, 0, 1)
    Image.fromarray(np.uint8(rgb*255)).convert('RGBA').save(path)


def add_eyes(g, binary, name, material_index=2, centers=None, radius=None):
    face = g['meshes'][0]['primitives'][0]
    p = accessor(g, binary, face['attributes']['POSITION'])
    uv = accessor(g, binary, face['attributes']['TEXCOORD_0'])
    tri = accessor(g, binary, face['indices']).reshape(-1, 3)
    joint_names = [g['nodes'][i]['name'] for i in g['skins'][0]['joints']]
    specs = []
    if centers is None:
        # UV-pole fallback: still force a symmetric, level pair (mirror about the mean x, same height, the less protruding depth)
        _c = {}
        for _side, _u in [('r', .39209), ('l', .60742)]:
            _pole = np.linalg.norm(uv-np.array([_u, .37476]), axis=1).argmin()
            _dup = np.where(np.linalg.norm(p-p[_pole], axis=1) < .0001)[0]
            _ring = np.unique(tri[np.isin(tri, _dup).any(axis=1)]); _ring = _ring[~np.isin(_ring, _dup)]
            _pts = p[_ring]; _c[_side] = np.array([_pts[:, 0].mean(), _pts[:, 1].mean()+.0003, _pts[:, 2].max()-.0117])
        _ax = (_c['l'][0]+_c['r'][0])/2; _hw = abs(_c['l'][0]-_c['r'][0])/2
        _y = (_c['l'][1]+_c['r'][1])/2; _z = min(_c['l'][2], _c['r'][2])
        centers = {'l': [_ax+_hw, _y, _z], 'r': [_ax-_hw, _y, _z]}
    for side, u in [('r', .39209), ('l', .60742)]:
        if centers is not None:
            center = np.asarray(centers[side], dtype=float)
        else:
            pole = np.linalg.norm(uv-np.array([u, .37476]), axis=1).argmin()
            pole_duplicates = np.where(np.linalg.norm(p-p[pole], axis=1) < .0001)[0]
            ring = np.unique(tri[np.isin(tri, pole_duplicates).any(axis=1)])
            ring = ring[~np.isin(ring, pole_duplicates)]
            points = p[ring]
            center = np.array([points[:, 0].mean(), points[:, 1].mean()+.0003, points[:, 2].max()-.0117])
        radius_xyz = np.array(radius if radius is not None else [.0115, .0108, .0108], dtype=float)
        latitude, longitude = 24, 32
        directions = np.array([[np.sin(np.pi*j/latitude)*np.cos(2*np.pi*i/longitude),
                                np.cos(np.pi*j/latitude),
                                np.sin(np.pi*j/latitude)*np.sin(2*np.pi*i/longitude)]
                               for j in range(latitude+1) for i in range(longitude+1)], dtype=float)
        positions = center + directions*radius_xyz
        normals = directions/radius_xyz; normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        eye_uv = np.c_[.5+directions[:, 0]*.5, .5-directions[:, 1]*.5]
        eye_tri = []
        for j in range(latitude):
            for i in range(longitude):
                a = j*(longitude+1)+i; b = a+longitude+1
                if j: eye_tri.append([a, a+1, b])
                if j < latitude-1: eye_tri.append([a+1, b+1, b])
        eye_tri = np.array(eye_tri)
        # Enforce outward orientation for the generated geometry too.
        geometric = np.cross(positions[eye_tri[:, 1]]-positions[eye_tri[:, 0]], positions[eye_tri[:, 2]]-positions[eye_tri[:, 0]])
        if np.sum(geometric*normals[eye_tri].mean(axis=1)) < 0: eye_tri = eye_tri[:, [0, 2, 1]]
        joints = np.zeros((len(positions), 4), dtype=np.uint16); joints[:, 0] = joint_names.index('skf_eye_'+side)
        weights = np.zeros((len(positions), 4)); weights[:, 0] = 1
        attrs = {kind: append_accessor(g, binary, values, typ, component) for kind, values, typ, component in
                 [('POSITION', positions, 'VEC3', 5126), ('NORMAL', normals, 'VEC3', 5126),
                  ('TEXCOORD_0', eye_uv, 'VEC2', 5126), ('JOINTS_0', joints, 'VEC4', 5123), ('WEIGHTS_0', weights, 'VEC4', 5126)]}
        g['meshes'][0]['primitives'].append({'attributes': attrs, 'indices': append_accessor(g, binary, eye_tri.reshape(-1, 1), 'SCALAR', 5123, 34963),
                                            'material': material_index, 'mode': 4})
        specs.append({'side': side, 'center': center.tolist(), 'radius': radius_xyz.tolist(), 'joint': 'skf_eye_'+side})
    return specs

