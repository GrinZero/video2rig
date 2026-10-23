"""Each painted limb's own outline in every frame, read from its colour. A clip that started from a painted start
frame (blender/render_markers.py) shows every limb in its own colour, so a pixel's colour says which limb it is: no
point tracker and no SAM 2, nothing to lose when four legs of the same fur cross, and a far leg hidden on frame 0
is found the moment it shows.

The colours are learned on the video's own frame 0: our start render, redrawn by the video model (its colours are
close to ours, not equal), where the pose is known and so is which pixel belongs to which bone. A bone whose paint
the video keeps too pale to tell from the fur (light paws, near-white toes) is left to the body: its limb is matched
down to where its colour ends, never stretched over the pale part.

Labels per view and frame, as parts.py: 0 nothing, 1 the body, 2 + i painted limb i (cached in
work/<clip>/parts-colour.npz).
"""
import json

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .parts import start_labels, triangle_objects

BINS = (32, 64, 64)          # L, a, b of OpenCV's 8-bit Lab
ERODE_PX = 2                 # frame-0 samples keep this far from a bone's edge (blur and the video's redraw)
WIN = 2.0                    # a pixel is a limb's when that limb's colour is this many times likelier than any other
READ_SHARE = .6              # a bone's paint is read when this share of its frame-0 pixels wins against the fur
MIN_AREA = 20                # px: smaller specks of a colour are dropped
MIN_SEEN = 20                # px of a bone on frame 0 to learn its colour there; fewer: predicted (hidden_colours)
FIT_SEEN = 200               # px of a bone on frame 0 for it to teach the paint -> video colour map


def painted_bones(cr, markers):
    m = json.loads(markers.read_text(encoding='utf-8'))
    names = cr.rig.names
    chains = []
    for chain, bones in m['limbs'].items():
        paint = [names.index(b) for b in bones if b in m['colours'] and b in names]
        if paint:
            chains.append((chain, paint))
    return chains


def owner(rig, roots):
    """Per bone: the first of `roots` met walking up from it (itself included), or -1."""
    out = np.full(rig.nb, -1)
    for j in range(rig.nb):
        p = j
        while p >= 0 and p not in roots:
            p = rig.parents[p]
        out[j] = p
    return out


def lab_of(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def bins_of(lab):
    return (lab[..., 0].astype(int) * BINS[0] // 256, lab[..., 1].astype(int) * BINS[1] // 256,
            lab[..., 2].astype(int) * BINS[2] // 256)


def chroma(lab):
    return np.hypot(lab[..., 1].astype(float) - 128, lab[..., 2].astype(float) - 128)


def paint_lab(rgb):
    """8-bit Lab of a markers.json colour (linear RGB) as a flat sRGB pixel."""
    r, g, b = [min(255.0, 255.0 * max(c, 0.0) ** (1 / 2.2)) for c in rgb]
    return cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def hidden_colours(samples, paint, hidden, rng):
    """Frame-0 samples for bones hidden on frame 0 (a far leg behind the near one): the video redraws every paint
    with the same shift of light and colour, so a map from paint to the video's colour is fitted on the bones it
    shows (affine in Lab) and applied to the hidden ones, spread like the seen bones' pixels."""
    seen = [j for j in samples if j >= 0 and j in paint and len(samples[j]) >= FIT_SEEN]
    if not seen or not hidden:
        return {}
    X = np.array([paint[j] for j in seen])
    Y = np.array([np.median(samples[j], 0) for j in seen])
    if len(seen) >= 5:
        A = np.c_[X, np.ones(len(X))]
        M, *_ = np.linalg.lstsq(A, Y, rcond=None)
        predict = lambda x: np.r_[x, 1.0] @ M
    else:
        shift = (Y - X).mean(0)
        predict = lambda x: x + shift
    spread = np.concatenate([samples[j] - np.median(samples[j], 0) for j in seen]).astype(float)
    out = {}
    for j in hidden:
        pick = spread[rng.choice(len(spread), min(3000, len(spread)), replace=False)]
        out[j] = np.clip(np.round(predict(paint[j]) + pick), 0, 255).astype(np.uint8)
    return out


def histogram(s):
    h = np.zeros(BINS, np.float32)
    np.add.at(h, tuple(b[0] for b in bins_of(s[None])), 1)
    h = gaussian_filter(h, 1.0)
    return h / max(h.sum(), 1e-9)


def segment(cr, clip, log=print):
    """(labels per view (T, H, W) uint8, bone objects, object names, info) or (None, ...) for a plain clip."""
    markers = cr.project.markers(clip)
    if markers is None or not markers.exists():
        return None, None, None, None
    rig = cr.rig
    chains = painted_bones(cr, markers)
    painted = [j for _, ps in chains for j in ps]
    work = cr.project.work(clip)
    path = work / 'parts-colour.npz'
    views = cr.views(clip)
    T = len(cr.times(clip))
    # frame 0: which pixel is which painted bone, through the aligned cameras
    of_bone = np.ones(rig.nb, int)
    up = owner(rig, set(painted))
    of_bone[up >= 0] = 2 + up[up >= 0]
    starts = start_labels(cr, clip, triangle_objects(cr, of_bone))
    k = np.ones((2 * ERODE_PX + 1,) * 2, np.uint8)
    samples = {}                                          # bone (or -1 the fur) -> Lab pixels of frame 0
    for vi, v in enumerate(views):
        lab0, sil = lab_of(v['frame'](0)), v['mask'](0)
        for o in np.unique(starts[vi]):
            if o < 1:
                continue
            m = cv2.erode((starts[vi] == o).astype(np.uint8), k).astype(bool) & sil
            key = -1 if o == 1 else int(o) - 2
            samples.setdefault(key, []).append(lab0[m])
    samples = {kk: np.concatenate(vv) for kk, vv in samples.items() if sum(len(x) for x in vv)}
    colours = json.loads(markers.read_text(encoding='utf-8'))['colours']
    paint = {j: paint_lab(colours[rig.names[j]]) for j in painted}
    hidden = [j for j in painted if len(samples.get(j, ())) < MIN_SEEN]
    predicted = hidden_colours(samples, paint, hidden, np.random.default_rng(0))
    samples.update(predicted)
    fur = samples.get(-1, np.zeros((0, 3), np.uint8))
    fur_h = histogram(fur)
    readable, info = set(), {'bones': {}}
    for j in painted:
        s = samples.get(j)
        share = 0.0
        if s is not None and len(s):
            # how many of the bone's own pixels its colour tells from the fur (light, darkness and hue together)
            b = tuple(x[0] for x in bins_of(s[None]))
            share = float((histogram(s)[b] > WIN * fur_h[b]).mean())
        ok = share >= READ_SHARE
        info['bones'][rig.names[j]] = {'apart_from_fur': round(share, 2), 'read': ok,
                                       'colour': 'predicted (hidden on frame 0)' if j in predicted else 'frame 0'}
        if ok:
            readable.add(j)
    # objects: each limb = its readable painted bones and what hangs below them (pale bones below a readable one stay
    # with the body: they are not seen, so they must not be pulled into the limb's colour)
    of = np.ones(rig.nb, int)
    names = ['body']
    pale = set(painted) - readable
    for chain, ps in chains:
        mine = [j for j in ps if j in readable]
        if not mine:
            continue
        names.append(chain)
        for j in range(rig.nb):
            p = j
            while p >= 0 and p not in mine and p not in pale:
                p = rig.parents[p]
            if p in mine:
                of[j] = len(names)
    info['objects'] = {n: [rig.names[j] for j in range(rig.nb) if of[j] == i + 1] for i, n in enumerate(names) if i}
    if path.exists():
        d = np.load(path, allow_pickle=True)
        if list(d['names']) == names and str(d['info']) == json.dumps(info):
            return [d[f'view{v}'] for v in range(len(views))], of, names, info
    # colour model: a smoothed Lab histogram per class (the fur, each limb, the pale bones as "not a limb")
    classes = [('fur', fur)]
    for i, (chain, ps) in enumerate(chains):
        if chain in names:
            s = [samples[j] for j in ps if j in readable and j in samples]
            classes.append((chain, np.concatenate(s)))
    if pale:
        classes.append(('pale', np.concatenate([samples[j] for j in pale if j in samples])))
    dens = np.stack([histogram(s) for _, s in classes])   # (classes, L, a, b)
    best = dens.argmax(0)
    second = np.sort(dens, 0)[-2]
    win = dens.max(0) > WIN * np.maximum(second, 1e-12)
    table = np.ones(BINS, np.uint8)                       # 1 = the body (fur, pale bones, undecided)
    for ci, (cname, _) in enumerate(classes):
        if cname in names:
            table[(best == ci) & win] = names.index(cname) + 1
    labels = []
    k3 = np.ones((3, 3), np.uint8)
    truth0 = start_labels(cr, clip, triangle_objects(cr, of))   # frame 0 by our own render: the check
    for vi, v in enumerate(views):
        H, W = v['mask'](0).shape
        lab_v = np.zeros((T, H, W), np.uint8)
        for t in range(T):
            sil = cv2.dilate(v['mask'](t).astype(np.uint8), k3).astype(bool)
            lab = table[bins_of(lab_of(v['frame'](t)))]
            lab[~sil] = 0
            for o in range(2, len(names) + 1):
                m = cv2.morphologyEx((lab == o).astype(np.uint8), cv2.MORPH_OPEN, k3)
                n, cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
                small = np.isin(cc, 1 + np.where(st[1:, cv2.CC_STAT_AREA] < MIN_AREA)[0])
                lab[(lab == o) & ((m == 0) | small)] = 1
            lab_v[t] = lab
        labels.append(lab_v)
        seen = {names[o - 1]: int(((lab_v == o).sum((1, 2)) >= 30).sum()) for o in range(2, len(names) + 1)}
        iou = {names[o - 1]: round(float(((lab_v[0] == o) & (truth0[vi] == o)).sum())
                                   / max(float(((lab_v[0] == o) | (truth0[vi] == o)).sum()), 1.0), 2)
               for o in range(2, len(names) + 1) if (truth0[vi] == o).sum() >= 30}
        log(f'colours: {v["name"]} view: frames where each limb shows {seen} of {T}; frame 0 against our render {iou}')
    np.savez_compressed(path, names=np.array(names), info=json.dumps(info),
                        **{f'view{v}': lab for v, lab in enumerate(labels)})
    return labels, of, names, info
