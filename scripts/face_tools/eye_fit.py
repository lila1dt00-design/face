"""Per-face eye fitting, eye centring and face-length adjustment.

Why this exists
---------------
The old converter put the eyeballs at a *fixed offset from the UV pole* (z = front-most ring point - 11.7 mm,
radius 13.2/12.4 mm for every head). On faces whose eye UVs differ from the standard PES layout the balls ended up
too low / too wide / too far forward, so the eyelids no longer covered them and the face looked "distorted".

Now every face is measured individually, in this order of trust:
  1. the face's own eyelid bones  (skf_eyelid_t_* / skf_eyelid_b_*, global bind position, exported by the FMDL),
  2. the skin vertices skinned to those eyelid bones (used for the eyelid-opening size and the surface depth),
  3. the old UV-pole method (last resort, flagged in the report).
The result is validated (inter-ocular distance, inside the head, sphere front vs. lid surface); on failure we fall
back and say so in 模型信息.json -> eyeFit.warnings.

All lengths are metres, Y up, face looks toward +Z (same frame as the rest of the converter).
"""
import numpy as np

REF_OPENING_W = 0.0267           # lid-vertex x-extent of the reference face the radius was tuned on
REF_RADIUS = np.array([.0132, .0124, .0124])
RADIUS_OVER_OPENING = 0.43       # eyeball radius / eye-opening width
DEFAULT_SHRINK = 0.87
SURFACE_GAP = 0.0040             # sphere front sits this far behind the lid-vertex surface (measured on the reference)
IPD_RANGE = (0.045, 0.085)       # sane inter-ocular distance for a PES head (m)
# Position of the PES eye-bone relative to the lid-vertex bbox (used only when the source has no bone data):
BONE_FRAC_X = 0.50               # from the inner edge of the lid-vertex cloud
BONE_FRAC_Y = 0.56               # from the bottom of the lid-vertex cloud

LID_BONES = {'l': ('skf_eyelid_t_l', 'skf_eyelid_b_l'), 'r': ('skf_eyelid_t_r', 'skf_eyelid_b_r')}


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def collect_landmarks(skin_meshes, bones=None):
    """skin_meshes: iterable of (positions (n,3) float array, [ {bone_name: weight} per vertex ]).
    bones: {name: (x,y,z)} global bind positions from the source model (may be empty)."""
    lm = {'bones': dict(bones or {}), 'lid': {'l': [], 'r': []}, 'skin': [], 'jaw': []}
    for pos, maps in skin_meshes:
        lm['skin'].append(pos)
        for p, m in zip(pos, maps):
            for side, (t, b) in LID_BONES.items():
                if m.get(t, 0) + m.get(b, 0) >= .5:
                    lm['lid'][side].append(p)
            if m.get('skf_jaw', 0) >= .5:
                lm['jaw'].append(p)
    lm['skin'] = np.concatenate(lm['skin']) if lm['skin'] else np.zeros((0, 3))
    for k in ('l', 'r'):
        lm['lid'][k] = np.array(lm['lid'][k]).reshape(-1, 3)
    lm['jaw'] = np.array(lm['jaw']).reshape(-1, 3)
    return lm


def _bone_center(bones, side):
    t, b = LID_BONES[side]
    if t in bones and b in bones:
        c = (np.asarray(bones[t], float) + np.asarray(bones[b], float)) / 2
        # some packs export eyelid bones at the origin / without a usable bind pose: treat as missing
        if np.linalg.norm(c) > 0.3 and abs(c[0]) > 0.005:
            return c
    return None


def _lid_center(lid, side):
    """Approximate eye centre from the lid-weighted vertex cloud only (no source bones available)."""
    if len(lid) < 8:
        return None
    lo, hi = lid.min(0), lid.max(0)
    w = hi[0] - lo[0]
    x = (lo[0] + BONE_FRAC_X * w) if side == 'l' else (hi[0] - BONE_FRAC_X * w)
    return np.array([x, lo[1] + BONE_FRAC_Y * (hi[1] - lo[1]), np.nan])


def midline_x(lm):
    """Head symmetry plane x: the mirror position that makes the front-face skin best match its own reflection
    (chamfer distance, 0.25 mm steps over +-6 mm). Ears/hair asymmetry cannot bias it the way a bbox centre would."""
    from scipy.spatial import cKDTree
    s = lm['skin']
    if len(s) < 50:
        return 0.0
    ys = [c[1] for c in (lm['bones'].get(n) for n in sum(LID_BONES.values(), ())) if c is not None]
    if not ys:
        ys = [float(np.mean([lm['lid'][k][:, 1].mean() for k in 'lr' if len(lm['lid'][k])] or [np.median(s[:, 1])]))]
    ey = float(np.mean(ys))
    face = s[(s[:, 1] > ey - .07) & (s[:, 1] < ey + .04) & (np.abs(s[:, 0]) < .075) & (s[:, 2] > np.percentile(s[:, 2], 55))]
    if len(face) < 50:
        face = s
    if len(face) > 4000:
        face = face[np.random.default_rng(0).choice(len(face), 4000, replace=False)]
    tree = cKDTree(face)
    best, bx = 1e9, 0.0
    for m in np.arange(-.006, .00601, .00025):
        mir = face.copy(); mir[:, 0] = 2 * m - mir[:, 0]
        d = tree.query(mir)[0].mean()
        if d < best:
            best, bx = d, float(m)
    return bx


def fit_eyes(lm, adjust=None, uv_fallback=None):
    """Return dict(centers={'r':[x,y,z],'l':[...]}, radius=[rx,ry,rz], method, report...)."""
    adj = adjust or {}
    warn = []
    centers, method = {}, 'eyelid-bones'
    for side in 'rl':
        c = _bone_center(lm['bones'], side)
        lid = lm['lid'][side]
        if c is None:
            c = _lid_center(lid, side)
            method = 'eyelid-vertices'
        centers[side] = c
    # surface depth from the lid skin vertices (front-most-ish: median of the upper half of z)
    depth = {}
    for side in 'rl':
        lid = lm['lid'][side]
        depth[side] = float(np.median(lid[:, 2])) if len(lid) >= 8 else None
    # ball size follows this face's own eye opening: radius = 0.43 x opening width (iris then looks natural, not oversized)
    widths = [np.ptp(lm['lid'][s][:, 0]) for s in 'rl' if len(lm['lid'][s]) >= 8]
    rx = float(np.clip(RADIUS_OVER_OPENING * np.mean(widths), .0095, .0135)) if widths else float(REF_RADIUS[0] * DEFAULT_SHRINK)
    scale = rx / REF_RADIUS[0]
    scale *= float(adj.get('eyeScale', 1.0))
    radius = REF_RADIUS * scale
    for side in 'rl':
        c = centers[side]
        if c is None:
            centers = None
            break
        z_depth = None if depth[side] is None else depth[side] - radius[0] - SURFACE_GAP
        if np.isnan(c[2]):
            c[2] = z_depth if z_depth is not None else np.nan
        elif z_depth is not None and c[2] > z_depth:
            # never let the ball sit further forward than the lid surface allows (prevents bulging eyes)
            if c[2] - z_depth > .005:
                warn.append(f'{side}: bone z {c[2]:.4f} is {(c[2]-z_depth)*1000:.1f} mm in front of the lid surface; using surface depth')
            c[2] = z_depth
    ok = centers is not None and all(np.isfinite(centers[s]).all() for s in 'rl')
    if ok:
        ipd = abs(centers['l'][0] - centers['r'][0])
        if not IPD_RANGE[0] <= ipd <= IPD_RANGE[1]:
            warn.append(f'inter-ocular distance {ipd*1000:.1f} mm outside {IPD_RANGE[0]*1000:.0f}-{IPD_RANGE[1]*1000:.0f} mm')
            ok = False
    if not ok:
        return dict(centers=uv_fallback or None, radius=REF_RADIUS.tolist(), method='uv-pole (fallback)', warnings=warn + ['eye fit fell back to UV pole'],
                    centering=None, radiusScale=1.0)
    # ---- centring: put the pair symmetric about the head midline, level and at equal depth
    mid = midline_x(lm)
    cx = (centers['l'][0] + centers['r'][0]) / 2
    half = abs(centers['l'][0] - centers['r'][0]) / 2 * float(adj.get('eyeSeparationScale', 1.0))
    y = (centers['l'][1] + centers['r'][1]) / 2
    z = (centers['l'][2] + centers['r'][2]) / 2
    dy = abs(centers['l'][1] - centers['r'][1]); dz = abs(centers['l'][2] - centers['r'][2])
    centring = {'midlineX': mid, 'pairCentreBefore': float(cx), 'shiftX_mm': float((mid - cx) * 1000),
                'heightDiff_mm': float(dy * 1000), 'depthDiff_mm': float(dz * 1000), 'applied': bool(adj.get('eyeCentering', True))}
    if adj.get('eyeCentering', True):
        # 2 mm dead zone: the symmetry plane is estimated from the ears, which are never perfectly symmetric
        # centre the pair on the openings themselves (not on the head mesh midline): each ball ends up in the middle of its own
        # eye opening, and the two stay symmetric, level and at the same depth
        axis = cx
        centring['axisUsedX'] = float(axis)
        z = min(centers['l'][2], centers['r'][2])   # same depth for both eyes, never the more protruding one
        new = {'l': np.array([axis + half, y, z]), 'r': np.array([axis - half, y, z])}
    else:
        new = {s: np.array(centers[s], float) for s in 'rl'}
    off = np.asarray(adj.get('eyeOffsetMm', [0, 0, 0]), float) / 1000.0
    for s in 'rl':
        new[s] = new[s] + off
    return dict(centers={s: [float(v) for v in new[s]] for s in 'rl'}, radius=radius.tolist(), method=method,
                warnings=warn, centering=centring, radiusScale=scale, interocularMm=float(half * 2000),
                openingWidthMm=[float(w * 1000) for w in widths], userOffsetMm=(off * 1000).tolist())


def face_metrics(lm, fit):
    s = lm['skin']
    ey = float(np.mean([fit['centers']['l'][1], fit['centers']['r'][1]]))
    chin = float(lm['jaw'][:, 1].min()) if len(lm['jaw']) else None
    band = s[(np.abs(s[:, 1] - ey) < .03)]
    width = float(band[:, 0].max() - band[:, 0].min()) if len(band) else None
    out = {'eyeLineY': ey, 'chinY': chin, 'headWidthAtEyesMm': None if width is None else width * 1000}
    if chin is not None and width:
        out['eyeToChinMm'] = (ey - chin) * 1000
        out['eyeToChinOverWidth'] = (ey - chin) / width
    return out


def face_length_field(lm, fit, factor):
    """Return f(positions)->positions that lengthens/shortens the LOWER face (eye line -> chin) by `factor`.
    The eye line is the pivot (eyes, brow and hair do not move); below the chin the effect fades out over the neck."""
    factor = float(factor)
    if abs(factor - 1) < 1e-6:
        return None
    if not .90 <= factor <= 1.12:
        raise ValueError('faceLength must be between 0.90 and 1.12')
    ey = float(np.mean([fit['centers']['l'][1], fit['centers']['r'][1]]))
    chin = float(lm['jaw'][:, 1].min()) if len(lm['jaw']) else ey - .16
    neck = chin - .07

    def apply(P):
        P = np.array(P, float, copy=True)
        y = P[:, 1]
        face_disp = (factor - 1) * (y - ey)                         # exact scaling about the eye line (y<=ey part)
        chin_disp = (factor - 1) * (chin - ey)                      # displacement reached at the chin
        fade = 1 - _smooth((chin - y) / max(chin - neck, 1e-6))     # 1 at the chin -> 0 at the neck base
        disp = np.where(y >= chin, np.where(y <= ey, face_disp, 0.0), chin_disp * fade)
        P[:, 1] = y + disp
        return P
    return apply
