"""Auto-rig for creatures on many legs (spiders, crabs, scorpions, insects): the body is the thick middle, every leg a
thin tube leaving it, traced from the body to its tip, joints where it bends.

    used by video2rig.autorig.unirig and video2rig.autorig.skeleton when six or more legs reach the floor:
    UniRig drew a spider's legs only to the knees, the centre lines of the whole body tangled where thin legs touch.

1. Voxels; the body = its thick core grown by its own depth (it takes in the crowded roots of the legs); the legs =
   the rest, in connected pieces.
2. In each piece, the path from where it touches the body to its furthest point, kept to the middle of the tube (a
   step costs more near the skin); two legs touching each other in one piece give two furthest points apart.
3. Six or more legs ending on the floor: a many-legged body. Legs too thin for the voxels (a spider's were 1-2
   voxels thick at 1% of the height) are traced again on voxels half the size.
4. Bones: the body along the creature's facing (abdomen behind = Tail1, middle = Hips, front = Head); four per leg,
   the joints at the strongest bends near a quarter, a half and three quarters of its length: Coxa, Femur, Tibia,
   Tarsus with the pair number front to back and .L/.R (no knee or elbow names: each joint folds the way the model
   already bends it).
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from . import skeleton as S

SEGMENTS = ('Coxa', 'Femur', 'Tibia', 'Tarsus')
OFFSETS = [np.array(d) for d in np.argwhere(np.ones((3, 3, 3))) - 1
           if tuple(d) > (0, 0, 0)]                                     # 13 forward neighbours


def split_body(solid, depth):
    """The body (its thick core, over half the body's own depth, grown by that depth again: it takes in the roots of
    the legs, where a spider's legs crowd together and joined into one piece with a slimmer body) and the leg
    pieces left over."""
    core = depth >= .5 * depth.max()
    body = (ndimage.distance_transform_edt(~core) <= 1.2 * depth.max()) & solid
    legs = solid & ~body
    lab, n = ndimage.label(legs, structure=np.ones((3, 3, 3)))
    return body, lab, n


def trace_piece(sl, lab_k, body_near, depth, lo, size, height):
    """Legs in one leg piece (bounding-box slice `sl`, mask `lab_k`): [(points from the body out to the tip)]."""
    idx = np.argwhere(lab_k)
    if len(idx) < 8:
        return []
    ids = -np.ones(lab_k.shape, np.int64)
    ids[tuple(idx.T)] = np.arange(len(idx))
    dep = depth[sl][tuple(idx.T)]
    rows, cols, w = [], [], []
    for d in OFFSETS:
        j = idx + d
        ok = np.all((j >= 0) & (j < np.array(lab_k.shape)), axis=1)
        a = np.nonzero(ok)[0]
        b = ids[tuple(j[ok].T)]
        keep = b >= 0
        a, b = a[keep], b[keep]
        step = float(np.linalg.norm(d))
        rows.append(a)
        cols.append(b)
        w.append(step * (1 + 3.0 / (.5 * (dep[a] + dep[b]) + .5)))      # dearer near the skin: keeps to the middle
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    G = coo_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                   shape=(len(idx), len(idx))).tocsr()
    src = np.nonzero(body_near[sl][tuple(idx.T)])[0]
    if not len(src):
        return []
    dist, pred, _ = dijkstra(G, indices=src, min_only=True, return_predecessors=True)
    pos = lo + (idx + np.array([s.start for s in sl]) + .5) * size
    legs, taken = [], np.zeros(len(idx), bool)
    for _ in range(4):
        cand = np.where(np.isfinite(dist) & ~taken, dist, -1)
        tip = int(np.argmax(cand))
        if cand[tip] <= 0:
            break
        path = [tip]
        while pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        P = pos[path[::-1]]
        length = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        if length < .3 * height:
            break
        legs.append(P)
        # the next leg in this piece lies away from this one (two legs touching mid-way or at their roots)
        d_path = np.min(np.linalg.norm(pos[:, None] - P[None, ::4], axis=2), axis=1)
        taken |= d_path < .12 * height
    return legs


def trace(verts, faces, height, log):
    """(legs, body points, voxel size): each leg the points from the body out to its tip."""
    best = None
    for frac in (.01, .005):
        size = frac * height
        solid, lo = S.voxelize(verts, faces, size)
        depth = ndimage.distance_transform_edt(solid).astype(np.float32)
        body, lab, n = split_body(solid, depth)
        near = ndimage.binary_dilation(body, structure=np.ones((3, 3, 3)))
        legs = []
        for k, sl in enumerate(ndimage.find_objects(lab)):
            if sl is None:
                continue
            legs += trace_piece(sl, lab[sl] == k + 1, near, depth, lo, size, height)
        thin = float(np.median(depth[lab > 0])) if n else 9.0
        best = (legs, lo + (np.argwhere(body) + .5) * size, size)
        log(f'autorig: many legs? {len(legs)} leg(s) traced on {frac * 100:.1f}% voxels '
            f'(legs {thin:.1f} voxels thick)')
        if thin >= 2.5:
            break
    return best


def build(work, log=print):
    """joints.json for a many-legged creature, or False when it does not stand on six or more legs."""
    work = Path(work)
    d = np.load(work / 'shell.npz')
    V, F = d['verts'].astype(np.float64), d['faces']
    h = float(d['height'])
    floor = float(V[:, 2].min())
    # quick look first: how many thin pieces of the shape reach the floor far from the body?
    solid, lo = S.voxelize(V, F, .01 * h)
    depth = ndimage.distance_transform_edt(solid)
    body, lab, n = split_body(solid, depth)
    reach = 0
    for k, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        idx = np.argwhere(lab[sl] == k + 1) + np.array([s.start for s in sl])
        P = lo + (idx + .5) * .01 * h
        if len(P) > 20 and P[:, 2].min() - floor < .06 * h and np.ptp(P, axis=0).max() > .25 * h:
            reach += 1
    if reach < 5:
        return False
    facing = S.find_facing(solid, lo, .01 * h, h, log=lambda m: None)
    legs, body_pts, size = trace(V, F, h, log)
    on_floor = [P for P in legs if P[-1, 2] - floor < .08 * h]
    if len(on_floor) < 6:
        return False
    fwd = S.AXES[facing]
    # the front is where the legs leave the body when that is clearly off its middle (a spider's abdomen has none);
    # in the middle (an insect's thorax) the shape's own guess stays
    ahead = (np.mean([P[0] for P in legs], axis=0) - body_pts.mean(0)) @ fwd
    if abs(ahead) > .1 * np.ptp(body_pts @ fwd) and ahead < 0:
        fwd = -fwd
        facing = ('+' if facing[0] == '-' else '-') + facing[1]
    left = np.cross([0, 0, 1.], fwd)
    vox = lo + (np.argwhere(solid) + .5) * .01 * h
    mid = float(np.median(vox @ left))
    # the body along the facing: rear end, a third, two thirds, front end (centres of slabs of the body)
    s = body_pts @ fwd
    a, b = np.percentile(s, 1), np.percentile(s, 99)

    def at(t):
        c = a + (b - a) * t
        sel = np.abs(s - c) < max(.04 * (b - a), 2 * size)
        return body_pts[sel].mean(0) if sel.any() else body_pts.mean(0)
    P0, P1, P2, P3 = at(.03), at(1 / 3), at(2 / 3), at(.97)
    bones = []

    def add(name, head, tail, parent, role):
        bones.append(dict(name=name, head=[float(v) for v in head], tail=[float(v) for v in tail], parent=parent,
                          role=role))
    root = np.array([P1[0], P1[1], floor])
    add('Root', root, root + [0, 0, .1 * h], None, 'root')
    add('Hips', P1, P2, 'Root', 'trunk')
    add('Tail1', P1, P0, 'Hips', 'tail')
    add('Head', P2, P3, 'Hips', 'head')
    trunk = {'Tail1': (P1, P0), 'Hips': (P1, P2), 'Head': (P2, P3)}

    def nearest_trunk(p):
        def dseg(h_, t_):
            u = np.clip((p - h_) @ (t_ - h_) / max((t_ - h_) @ (t_ - h_), 1e-12), 0, 1)
            return np.linalg.norm(p - (h_ + u * (t_ - h_)))
        return min(trunk, key=lambda k: dseg(*trunk[k]))
    win = max(2, int(round(.035 * h / size)))
    sides = {1: [], -1: []}
    for P in legs:
        sides[1 if P[0] @ left - mid > 0 else -1].append(P)
    paths = {side: [S.smooth(S.resample(P, size)[0], 1.5) for P in sorted(group, key=lambda P: -(P[0] @ fwd))]
             for side, group in sides.items()}                                     # front pair first

    def fractions(Ps):
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Ps, axis=0), axis=1))])
        return [float(s[int(np.argmin(np.linalg.norm(Ps - j, axis=1)))] / s[-1]) for j in S.place(Ps, [.22, .5, .78], win)]

    def at_fraction(Ps, f):
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Ps, axis=0), axis=1))])
        return np.array([np.interp(f * s[-1], s, Ps[:, c]) for c in range(3)])
    for side in sides:
        sfx = S.SIDES[side]
        for k, Ps in enumerate(paths[side]):
            # twins get their joints at the same share of their length (the mean of where each bends most)
            fr = fractions(Ps)
            twin = paths[-side][k] if k < len(paths[-side]) else None
            if twin is not None:
                fr = list((np.array(fr) + np.array(fractions(twin))) / 2)
            joints = [Ps[0]] + [at_fraction(Ps, f) for f in fr] + [Ps[-1]]
            parent = nearest_trunk(Ps[0])
            for seg, (j0, j1) in zip(SEGMENTS, zip(joints, joints[1:])):
                name = f'{seg}{k + 1}{sfx}'
                add(name, j0, j1, parent, 'chain')
                parent = name
    (work / 'joints.json').write_text(json.dumps({'body': 'multileg', 'facing': facing, 'height': h,
                                                  'left': [float(v) for v in left], 'source': 'many legs',
                                                  'bones': bones}, indent=1) + '\n', encoding='utf-8')
    log(f'autorig: many legs: {len(legs)} legs traced from the body ({len(on_floor)} on the floor), '
        f'{len(sides[1])} left, {len(sides[-1])} right')
    return True
