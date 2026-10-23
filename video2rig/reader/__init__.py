"""The joint reader: a small network that finds every joint of ONE creature in a Seedance frame, trained on renders of
that creature's own rig in random poses through the clip's own cameras (every label exact, nothing drawn by hand).

    python -m video2rig.reader.poses <project>      random poses within the rig's ranges -> <project>/reader/poses.npz
    blender model.blend -b --python blender/render_poses.py -- ...   the pictures and the true joints
    python -m video2rig.reader.train <project>      the reader -> <project>/reader/reader.pt
    python -m video2rig.reader.read <project> <clip>   every joint in every frame and panel

Keypoints are the bones' heads (the joints) plus the tips of bones that end free (toes, tail, ears, jaw, snout),
points off the axis of the limbs' last bones and of the head (the outermost claws, the antlers' ends: side_points),
and a few points of every prop (a log's two ends and middle, a stone's middle and its sides): (label, '@' + prop,
[x, y, z] in armature space at the start), the prop moving them rigidly.
"""
import re

import numpy as np

TWINS = (('.L', '.R'), ('_L', '_R'), ('.l', '.r'), ('_l', '_r'), ('Left', 'Right'), ('left', 'right'))
SIDE_SPAN = .6       # a bone whose skin spreads across its axis by at least this share of its length gets side points
SIDE_REACH = .22     # and each side reaches at least this share of the length from the axis


def twin_label(label):
    """The label on the other side: .L/.R, _L/_R, Left/Right (Mixamo), or None."""
    for a, b in TWINS:
        for x, y in ((a, b), (b, a)):
            if x in label:
                return label.replace(x, y, 1)
    return None


def side_of(label):
    """'L', 'R' or None."""
    for a, b in TWINS:
        if a in label:
            return 'L'
        if b in label:
            return 'R'
    return None


def keypoints(cr):
    """(label, bone name, f) of every point the reader finds, at head + f (tail - head) of its bone: each bone's head
    (f = 0) below the rig's root bone (a "Root" above the hips carries no joint of the body), and the free end of
    every bone whose tail no child starts from (a toe, the tail's end, an ear, the jaw, the snout). A free end sits
    where the bone's own skin ends along it, not at the bone's tail: a wolf's toe bones end 1-3 cm past the paw and
    its tail bone 5 cm past the fur, and a point in the air is one no picture shows."""
    from ..creature import FINGER
    rig = cr.rig
    p = np.zeros(rig.n)
    H, T = rig.bone_points(p)
    X = rig.skin(p, cr.mesh['coords'], cr.idx, cr.wts)
    above = {j for j in range(rig.nb) if j != rig.root and rig.desc[j, rig.root]}
    # fingers, eyes and end bones: too small for a 720p panel (a jaw stays: a bite opens it)
    above |= {j for j in range(rig.nb) if FINGER.search(rig.names[j]) and not re.search('jaw', rig.names[j], re.I)}
    out = []
    for j in range(rig.nb):
        if j not in above:
            out.append((rig.names[j], rig.names[j], 0.0))
    for j in range(rig.nb):
        if j in above:
            continue
        kids = [k for k in range(rig.nb) if rig.parents[k] == j and k not in above]
        if any(np.linalg.norm(H[k] - T[j]) <= .02 * cr.height for k in kids):
            continue
        w = np.where(cr.idx == j, cr.wts, 0).sum(1)
        mine = X[w > .3]
        L = np.linalg.norm(T[j] - H[j])
        if len(mine) < 10 or L < 1e-6:
            continue
        f = float(np.percentile((mine - H[j]) @ (T[j] - H[j]) / L ** 2, 98))
        if f > .25:
            out.append((rig.names[j] + '.tip', rig.names[j], round(min(f, 1.0), 3)))
    return out + side_points(cr) + prop_points(cr)


def side_points(cr):
    """Points off the bones' axes, on the last bone of every limb chain (toes, a hand, a hoof, the head at the end of
    the neck) where the skin spreads across the bone (claws fanning out, antlers, a beak): the two skin vertices
    farthest out along the direction the skin spreads most, and along the one across it when it spreads there too.
    Every other keypoint lies on its bone's line, so nothing read says how the bone is turned about itself (a hand
    read with the palm backwards, a head with its antlers wrong); two points off the line give the whole turn.
    (label, bone, [x, y, z]): the point's rest position (the mesh's own coordinates), riding rigidly on its bone.
    Labels are mirror-consistent: 1 is the outward (left side's left, right side's right), forward or upward one."""
    from ..creature import FINGER
    rig = cr.rig
    coords = np.asarray(cr.mesh['coords'], float)
    lateral = np.cross(cr.up, cr.facing)
    lateral /= max(np.linalg.norm(lateral), 1e-9)
    out = []
    for ch in cr.limb_chains(as_list=True):
        j = ch[-1]
        name = rig.names[j]
        if FINGER.search(name):
            continue
        mine = (cr.idx[:, 0] == j) & (cr.wts[:, 0] >= .7)
        if mine.sum() < 20:
            continue
        X = coords[mine]
        h, u, L = rig.rest_head[j], rig.rest_rot[j][:, 1], float(rig.length[j])
        if L < 1e-6:
            continue
        d = X - h
        perp = d - np.outer(d @ u, u)
        w, E = np.linalg.eigh(np.cov(perp.T))
        E = E[:, ::-1]                                                  # the direction the skin spreads most first
        outward = lateral * (-1.0 if side_of(name) == 'R' else 1.0)
        for a, tag in ((0, 'a'), (1, 'b')):
            e = E[:, a] - (E[:, a] @ u) * u
            if np.linalg.norm(e) < 1e-6:
                continue
            e /= np.linalg.norm(e)
            ref = max((outward, cr.facing, cr.up), key=lambda r: abs(float(r @ e)))
            if float(ref @ e) < 0:
                e = -e
            s = perp @ e
            lo, hi = float(np.percentile(s, 1)), float(np.percentile(s, 99))
            if hi - lo < SIDE_SPAN * L or hi < SIDE_REACH * L or -lo < SIDE_REACH * L:
                continue
            for k, (val, n) in enumerate(((hi, 1), (lo, 2))):
                i = int(np.argmin(np.abs(s - val)))
                out.append((f'{name}.side{tag}{n}', name, [round(float(v), 5) for v in X[i]]))
    return out


def is_offaxis(kp):
    """A keypoint given by its rest position (side_points), not by a place along its bone."""
    return not is_prop(kp) and not np.isscalar(kp[2])


def along(kp, rig):
    """Where a keypoint lies along its bone, as a share of the bone's length (for ordering points down a limb)."""
    if is_prop(kp):
        return 0.0
    if not is_offaxis(kp):
        return float(kp[2])
    j = rig.names.index(kp[1])
    return float((np.asarray(kp[2], float) - rig.rest_head[j]) @ rig.rest_rot[j][:, 1] / max(float(rig.length[j]), 1e-9))


def prop_points(cr):
    """Points of every prop the reader can find: a long prop (a log) both ends of its longest axis and the middle; a
    round one (a stone) its middle only. A stone's "ends" and "sides" are points on a surface that looks the same all
    round: the reader named them by guess and was sure and wrong on 32-46% of them (dev/DECISION-2026-10-11.md)."""
    rig = cr.rig
    out = []
    for k, pr in enumerate(rig.props):
        X = cr.mesh['coords'][cr.idx[:, 0] == rig.nb + k]
        if len(X) < 4:
            continue
        c = X.mean(0)
        _, E = np.linalg.eigh(np.cov((X - c).T))
        E = E[:, ::-1]                                              # longest axis first
        proj = (X - c) @ E
        ext = np.abs(proj).max(0)
        pts = [('mid', c)]
        if ext[1] <= .5 * ext[0]:                                   # long: its ends tell its turn
            pts = [('end1', c + E[:, 0] * proj[:, 0].max()), ('end2', c + E[:, 0] * proj[:, 0].min()), ('mid', c)]
        out += [(f"{pr['name']}.{nm}", '@' + pr['name'], [round(float(v), 5) for v in x]) for nm, x in pts]
    return out


def is_prop(kp):
    return str(kp[1]).startswith('@')


def end_pairs(kps):
    """(k1, k2) of every prop's two ends: a log's ends look alike, so in every picture the reader names them by
    where they are (the one further left in that picture first) and the fit decides which 3D end each one is."""
    labels = [k[0] for k in kps]
    out = []
    for k, lb in enumerate(labels):
        if is_prop(kps[k]) and lb.endswith('.end1') and lb[:-1] + '2' in labels:
            out.append((k, labels.index(lb[:-1] + '2')))
    return out


def points(kps, names, heads, tails, props=None, prop_names=(), rots=None, rig=None):
    """(..., K, 3) keypoint positions from bone heads and tails (..., B, 3) listed in the order of `names`, and the
    props' matrices (..., P, 4, 4) in the order of `prop_names` (armature space). A point off its bone's axis
    (side_points) needs the bones' world rotations too: rots (..., B, 3, 3) in the order of `names`, and the rig."""
    col = {n: i for i, n in enumerate(names)}
    heads, tails = np.asarray(heads), np.asarray(tails)
    out = np.zeros(heads.shape[:-2] + (len(kps), 3))
    prop_names = [str(n) for n in prop_names]
    for k, kp in enumerate(kps):
        if is_prop(kp):
            M = np.asarray(props)[..., prop_names.index(kp[1][1:]), :, :]
            out[..., k, :] = M[..., :3, :3] @ np.asarray(kp[2], float) + M[..., :3, 3]
        elif is_offaxis(kp):
            if rots is None or rig is None:
                raise ValueError(f'{kp[0]} lies off its bone: points() needs the bones\' rotations (rots, rig)')
            i, j = col[kp[1]], rig.names.index(kp[1])
            local = rig.rest_rot[j].T @ (np.asarray(kp[2], float) - rig.rest_head[j])
            out[..., k, :] = heads[..., i, :] + np.asarray(rots)[..., i, :, :] @ local
        else:
            i = col[kp[1]]
            out[..., k, :] = heads[..., i, :] + float(kp[2]) * (tails[..., i, :] - heads[..., i, :])
    return out


def render_params(cr, poses, lab):
    """The rig's parameters of every rendered training picture (reader/poses.npz and renders/labels.npz): the
    bones' rotations as Blender rendered them (with motion blur the pose at the middle of the shutter: the mean of
    the two keys' quaternions, normalised), the root put where the render's joints are. (N, rig.n) in the order of
    labels.npz; the pose then reproduces the render's joints to a hundredth of a centimetre."""
    from scipy.spatial.transform import Rotation
    rig = cr.rig
    bones = [str(b) for b in poses['bones']]
    names = [str(n) for n in lab['names']]
    by_bone = [names.index(b) for b in bones]
    Q = (poses['quats_a'] + poses['quats_b']) / 2 if 'quats_a' in poses.files else poses['quats']
    out = np.zeros((len(lab['index']), rig.n))
    for k, i in enumerate(lab['index']):
        q = Q[int(i)] / np.linalg.norm(Q[int(i)], axis=1, keepdims=True)
        basis = np.tile(np.eye(4), (len(bones), 1, 1))
        basis[:, :3, :3] = Rotation.from_quat(q, scalar_first=True).as_matrix()
        out[k] = rig.params_from_basis(basis, bones, heads=np.asarray(lab['heads'][k], float)[by_bone])
    return out


def render_keypoints(cr, kps, lab, poses_path, cache=None):
    """(N, K, 3) every keypoint's position in every rendered training picture (labels.npz order): the points on the
    bones from the render's own joints, the points off them (side_points) and the props from the full pose
    (render_params, cached as `cache` .npy)."""
    names = [str(n) for n in lab['names']]
    off = [k for k, kp in enumerate(kps) if is_offaxis(kp)]
    on = [kp if not is_offaxis(kp) else (kp[0], kp[1], 0.0) for kp in kps]
    X = points(on, names, lab['heads'], lab['tails'], lab['prop_mats'] if 'prop_mats' in lab.files else None,
               lab['prop_names'] if 'prop_names' in lab.files else ())
    if off:
        params = None
        if cache is not None and cache.exists():
            params = np.load(cache)
            if len(params) != len(lab['index']) or params.shape[1] != cr.rig.n:
                params = None
        if params is None:
            params = render_params(cr, np.load(poses_path), lab)
            if cache is not None:
                np.save(cache, params)
        kr = KeypointRig(cr, kps)
        for n in range(len(X)):
            X[n, off] = kr.points(params[n])[off]
    return X


def project_ortho(rec, arm_world, X, size):
    """Pixels (continuous, column j covers [j, j + 1)) of armature-space points X in a picture of `size` (w, h)
    rendered through camera `rec` of cameras.json (vertical sensor fit: Blender's render). Orthographic, or with
    perspective (type 'persp', focal_px at the camera's own height_px)."""
    M = np.asarray(arm_world, float)
    Xw = X @ M[:3, :3].T + M[:3, 3]
    d = Xw - np.asarray(rec['location'], float)
    right, up = np.asarray(rec['right'], float), np.asarray(rec['up'], float)
    if rec['type'] == 'ortho':
        k = size[1] / rec['ortho_height']
        u = d @ right * k + size[0] / 2
        v = -(d @ up) * k + size[1] / 2
        return np.stack([u, v], -1)
    f = size[1] * float(rec['focal_px']) / float(rec['height_px'])
    z = np.maximum(d @ np.asarray(rec['forward'], float), 1e-6)
    u = (d @ right) / z * f + size[0] / 2
    v = -(d @ up) / z * f + size[1] / 2
    return np.stack([u, v], -1)


project_view = project_ortho


class KeypointRig:
    """The keypoints as points riding on their bones: positions (and their Jacobian) for rig parameters."""

    def __init__(self, cr, kps):
        rig = cr.rig
        self.rig = rig
        idx = np.zeros((len(kps), 4), np.int64)
        wts = np.zeros((len(kps), 4))
        rest = np.zeros((len(kps), 3))
        names = [pr['name'] for pr in rig.props]
        for k, (_, bone, f) in enumerate(kps):
            if str(bone).startswith('@'):                  # a prop's point: carried rigidly by the prop
                rest[k] = np.asarray(f, float)
                idx[k, 0], wts[k, 0] = rig.nb + names.index(bone[1:]), 1.0
                continue
            j = rig.names.index(bone)
            if not np.isscalar(f):                         # a point off the bone's axis: its rest position
                rest[k] = np.asarray(f, float)
                idx[k, 0], wts[k, 0] = j, 1.0
                continue
            head = rig.rest_head[j]
            tail = head + rig.rest_rot[j][:, 1] * rig.length[j]
            rest[k] = head + f * (tail - head)
            idx[k, 0], wts[k, 0] = j, 1.0
        self.rest, self.idx, self.wts = rest, idx, wts

    def points(self, p, jac=False):
        return self.rig.skin(p, self.rest, self.idx, self.wts, jac=jac)


def tile_sizes(cams, fit, height=720):
    """Each panel's training picture size, as blender/render_poses.py renders it: the panel at the clip's frame
    height, letterboxed into a fit x fit square."""
    clip_scale = height / cams['frame_px'][1]
    out = []
    for rec in cams['views']:
        cw, ch = rec['width_px'] * clip_scale, rec['height_px'] * clip_scale
        s = min(fit / cw, fit / ch)
        h = int(round(ch * s))
        out.append((int(round(cw * h / ch)), h))
    return out
