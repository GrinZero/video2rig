"""Auto-rig, step 2: the skeleton of any creature from its watertight shell (blender/autorig_prepare.py).

    python -m video2rig.autorig.skeleton <work folder> [--body auto|quadruped|biped|centaur|winged]
        -> <work>/joints.json (bones: name, head, tail, parent, role) and <work>/skeleton-sheet.png

1. The shell filled into voxels (~1/100 of the height), thinned to its centre lines (skimage), made a tree.
2. Ends of the tree sorted by where they are: FEET on the floor, the HEAD front and top on the middle plane, the
   TAIL at the back, the rest LIMBS (arms, wings) in mirrored pairs.
3. The SPINE is the tree path from the tail (or the pelvis) to the head; every limb hangs from it where its own path
   meets it. Hind legs = the pair that meets furthest back.
4. Joints along each path where it bends most, near the place a body of that kind has them (a knee about a third
   down a horse's hind leg); bones named the way the tool reads them (Shin/Forearm bend one way, .L/.R twins).

Up is +Z, the creature faces the shell's `facing` (Tripo: -Y), its left is up x forward.
"""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import ndimage

SIDES = {1: '.L', -1: '.R'}


# ----- voxels and centre lines --------------------------------------------------------------------------------
def voxelize(V, F, size):
    lo = V.min(0) - 2 * size
    dims = np.ceil((V.max(0) + 2 * size - lo) / size).astype(int) + 1
    grid = np.zeros(dims, bool)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    longest = np.maximum(np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1)),
                         np.linalg.norm(c - b, axis=1))
    n = np.clip(np.ceil(longest / (size * .5)).astype(int), 1, 64)
    for k in np.unique(n):
        sel = n == k
        u, v = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing='ij')
        keep = (u + v) <= k
        u, v = u[keep] / k, v[keep] / k
        P = (a[sel, None] * (1 - u - v)[None, :, None] + b[sel, None] * u[None, :, None]
             + c[sel, None] * v[None, :, None]).reshape(-1, 3)
        ijk = np.floor((P - lo) / size).astype(int)
        grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    solid = ndimage.binary_fill_holes(grid)
    return solid, lo


def centre_tree(solid, lo, size):
    from skimage.morphology import skeletonize
    dist = ndimage.distance_transform_edt(solid) * size
    sk = skeletonize(solid)
    pts = np.argwhere(sk)
    index = {tuple(p): i for i, p in enumerate(pts)}
    G = nx.Graph()
    for i, p in enumerate(pts):
        G.add_node(i, pos=lo + (p + .5) * size, r=float(dist[tuple(p)]))
    for i, p in enumerate(pts):
        for d in np.argwhere(np.ones((3, 3, 3))) - 1:
            if not d.any():
                continue
            j = index.get(tuple(p + d))
            if j is not None and j > i:
                G.add_edge(i, j, w=float(np.linalg.norm(d)) * size)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.minimum_spanning_tree(G, weight='w')


def prune(T, min_len):
    """Spurs shorter than min_len cut off, again and again (a spur's removal can leave another)."""
    T = T.copy()
    changed = True
    while changed:
        changed = False
        for leaf in [n for n in T.nodes if T.degree(n) == 1]:
            if leaf not in T or T.degree(leaf) != 1:
                continue
            path, length, n = [leaf], 0.0, leaf
            while True:
                nxt = [m for m in T.neighbors(n) if m not in path]
                if len(nxt) != 1:
                    break
                length += T.edges[n, nxt[0]]['w']
                n = nxt[0]
                if T.degree(n) != 2:
                    break
                path.append(n)
            if T.degree(n) >= 3 and length < min_len:
                T.remove_nodes_from(path)
                changed = True
    return T


def collapse_tips(T, max_len):
    """Fingers, claws and toes: where two or more short end branches leave the same fork (a hand, a foot), only the
    longest stays, so the hand is one limb end, not four limbs (the troll's fingers came out as four arm pairs)."""
    T = T.copy()
    changed = True
    while changed:
        changed = False
        forks = {}
        for leaf in [n for n in T.nodes if T.degree(n) == 1]:
            path, length, prev, n = [leaf], 0.0, None, leaf
            while True:
                nxt = [m for m in T.neighbors(n) if m != prev]
                if not nxt:
                    break
                length += T.edges[n, nxt[0]]['w']
                prev, n = n, nxt[0]
                if T.degree(n) >= 3:
                    break
                path.append(n)
            if T.degree(n) >= 3:
                forks.setdefault(n, []).append((length, path))
        for fork, branches in forks.items():
            short = sorted([b for b in branches if b[0] < max_len], key=lambda b: -b[0])
            if len(short) >= 2 and T.degree(fork) - len(short) <= 1:
                for _, path in short[1:]:
                    T.remove_nodes_from(path)
                changed = True
                break
    return T


def pos(T, n):
    return T.nodes[n]['pos']


def path_points(T, path):
    return np.array([pos(T, n) for n in path])


def resample(P, step):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return P[:1], s[:1]
    t = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1), t


def smooth(P, sigma):
    if len(P) < 3:
        return P
    out = ndimage.gaussian_filter1d(P, sigma, axis=0, mode='nearest')
    out[0], out[-1] = P[0], P[-1]
    return out


def bends(P, window):
    """Turning angle (degrees) at every point of a resampled path, over +-window points."""
    ang = np.zeros(len(P))
    for i in range(window, len(P) - window):
        u, v = P[i] - P[i - window], P[i + window] - P[i]
        c = u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)
        ang[i] = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return ang


def place(P, fractions, window, reach=.12):
    """Joint positions along path P (start to end) near the given fractions of its length: the strongest bend
    within +-reach of each, else the fraction itself."""
    _, s = resample(P, 1e9) if len(P) < 2 else (None, None)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    ang = bends(P, window)
    out = []
    used = []
    for f in fractions:
        near = np.nonzero(np.abs(s / max(L, 1e-9) - f) <= reach)[0]
        near = [i for i in near if all(abs(i - u) > window for u in used)]
        if near and ang[near].max() > 12:
            i = near[int(np.argmax(ang[near]))]
        else:
            i = int(np.argmin(np.abs(s - f * L)))
        used.append(i)
        out.append(P[i])
    return out


# ----- the body -----------------------------------------------------------------------------------------------
LEG_PLANS = {
    # (a bone from the spine to the limb's root joint or None, the limb's bones, joint fractions along the path from
    # the root joint, where the limb leaves the trunk, to the end)
    'hind': (None, ['Thigh', 'Shin', 'HindPaw', 'HindToes'], [.32, .66, .88]),
    'front': ('Scapula', ['UpperArm', 'Forearm', 'FrontPaw', 'FrontToes'], [.30, .66, .88]),
    'biped': (None, ['Thigh', 'Shin', 'Foot', 'Toes'], [.45, .82, .92]),
    'arm': ('Shoulder', ['Arm', 'LowerArm', 'Hand'], [.42, .80]),   # not UpperArm/Forearm: a centaur has both
    'wing': ('WingShoulder', ['WingUpper', 'WingForearm', 'WingHand'], [.30, .62]),
}


def limb_path(T, path, step):
    """A limb's centre line resampled and smoothed, with the body's radius (thickness) at every point."""
    raw = path_points(T, path)
    s_raw = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))])
    P, t = resample(raw, step)
    r = np.interp(t, s_raw, [T.nodes[n]['r'] for n in path])
    return smooth(P, 1.5), r


def cut_held_prop(P, r, height, voxel, log, name):
    """An arm whose path runs on into something as thin as the voxel grid can show for over a tenth of the height
    holds a prop (a sword, a spear): the arm ends a little past where the thin part begins (the fist), the prop is not
    bone. The centaur's sword moved its elbow to the middle of the forearm and its hand onto the blade; fingers and
    claws (the alien's) are thin only for a few hundredths of the height."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    body = float(np.median(r[: max(3, int(.7 * len(r)))]))
    thin = r <= min(1.5 * voxel, .5 * body)
    k = len(r)
    while k > 0 and thin[k - 1]:
        k -= 1
    if k >= len(r) or s[-1] - s[k] < .10 * height:
        return P, r
    j = min(len(P) - 1, int(np.searchsorted(s, s[k] + .03 * height)))
    log(f'autorig: {name} holds a prop {s[-1] - s[k]:.2f} x height long: the arm ends at its fist')
    return P[:j + 1], r[:j + 1]


def extend_to(P, length, height):
    """A limb path made as long as its twin along its own last direction (an arm resting on the body fused with it
    in the voxels and its path stopped at the forearm)."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    missing = length - s[-1]
    if missing <= 0:
        return P
    i = int(np.searchsorted(s, s[-1] - .04 * height))
    d = P[-1] - P[min(i, len(P) - 2)]
    d = d / max(np.linalg.norm(d), 1e-9)
    n = max(2, int(np.ceil(missing / (.01 * height))))
    return np.concatenate([P, P[-1] + d[None] * np.linspace(0, missing, n + 1)[1:, None]])


def exit_index(P, axis_pts, radius, k=.75):
    """The first point of path P (spine side first) further than k x radius from the spine line: where the limb
    leaves the trunk (a hip socket, a shoulder joint)."""
    d = np.min(np.linalg.norm(P[:, None] - axis_pts[None], axis=2), axis=1)
    out = np.nonzero(d > k * radius)[0]
    return int(out[0]) if len(out) else 0


AXES = {'-Y': np.array([0, -1., 0]), '+Y': np.array([0, 1., 0]), '-X': np.array([-1., 0, 0]),
        '+X': np.array([1., 0, 0])}


def find_facing(solid, lo, size, height, log=print):
    """Which way the creature faces, from its shape alone: the sideways axis is the one it is mirror-symmetric
    about; forward is the side where its head sits higher and its toes reach (a Tripo model faced -Y once and +-X
    another time)."""
    idx = np.argwhere(solid)
    P = lo + (idx + .5) * size
    c = P.mean(0)
    # mirrored voxels landing within 1.5 voxels of the body, the plane moved a little round the centre for the best
    # fit: exact voxel overlap scored a hunched giant 0.68 both ways and took its front-to-back axis for its sides
    near = ndimage.distance_transform_edt(~solid)
    sym = {}
    for ax in (0, 1):
        best = 0.0
        for off in np.linspace(-.04, .04, 9) * np.ptp(P[:, ax]):
            m = P.copy()
            m[:, ax] = 2 * (c[ax] + off) - m[:, ax]
            ijk = np.floor((m - lo) / size).astype(int)
            ok = np.all((ijk >= 0) & (ijk < np.array(solid.shape)), axis=1)
            d = np.full(len(P), 99.0)
            d[ok] = near[tuple(ijk[ok].T)]
            best = max(best, float((d <= 1.5).mean()))
        sym[ax] = best
    side = max(sym, key=sym.get)                     # mirror-symmetric across it: the creature's sideways axis
    f = 1 - side                                     # the other horizontal axis runs front to back
    z = P[:, 2] - P[:, 2].min()
    mid = np.abs(P[:, side] - c[side]) < .06 * height
    top = mid & (z > np.percentile(z[mid], 97))
    head = (P[top, f].mean() - c[f]) / height if top.any() else 0.0
    low, ankle = z < .04 * height, (z > .06 * height) & (z < .15 * height)
    toes = (P[low, f].mean() - P[ankle, f].mean()) / height if low.any() and ankle.any() else 0.0
    score = head + 3 * toes
    facing = ('+' if score > 0 else '-') + 'XY'[f]
    log(f'autorig: facing {facing} (mirror symmetry x {sym[0]:.2f} y {sym[1]:.2f}; head {head:+.3f}, toes {toes:+.3f})')
    return facing


def analyse(T, height, facing, body='auto', log=print, mid=0.0, mid_tol=.06):
    fwd = AXES[facing]
    up = np.array([0, 0, 1.])
    left = np.cross(up, fwd)
    # left and right of the body's own middle (a sword held out sideways moved the frame's centre 14 cm off it)
    lat = lambda p: p @ left - mid
    ends = [n for n in T.nodes if T.degree(n) == 1]
    P = {n: pos(T, n) for n in ends}
    floor = min(p[2] for p in P.values())
    feet = [n for n in ends if P[n][2] - floor < .08 * height]
    # a long tail lying on the floor is not a foot: on the middle plane, far behind the feet's middle
    if len(feet) > 2:
        back = np.mean([P[n] @ fwd for n in feet])
        feet = [n for n in feet if not (abs(lat(P[n])) < .1 * height and P[n] @ fwd < back - .25 * height)]
    rest = [n for n in ends if n not in feet]
    # the head end first, from every end: the most forward end on the middle plane (the snout or the face), not an ear
    # or horn tip beside it; a head sunk between the shoulders (a hunched giant) has no end of its own: the highest
    # middle point of the tree then
    middle = [n for n in rest if abs(lat(P[n])) < mid_tol * height]
    if middle:
        head = max(middle, key=lambda n: P[n] @ fwd + .25 * P[n] @ up)
    else:
        mids = [n for n in T.nodes if abs(lat(pos(T, n))) < .06 * height]
        head = max(mids, key=lambda n: pos(T, n) @ up + .25 * pos(T, n) @ fwd)
        P[head] = pos(T, head)
    tails = [n for n in rest if n != head and abs(lat(P[n])) < .2 * height and P[n] @ fwd < np.mean(
        [P[m] @ fwd for m in ends]) - .15 * height]
    tail = min(tails, key=lambda n: P[n] @ fwd) if tails else None
    # horns and ears belong to the head; the rest: ends close together are one hand or one foot (fingers, claws,
    # toes forking one after another), the end furthest from the body's middle stands for them
    rest = [n for n in rest if n not in (head, tail) and np.linalg.norm(P[n] - P[head]) > .15 * height]
    centre = np.mean([pos(T, n) for n in T.nodes], axis=0)
    groups = []
    for n in sorted(rest, key=lambda n: -np.linalg.norm(P[n] - centre)):
        for g in groups:
            if np.linalg.norm(P[n] - P[g[0]]) < .15 * height:
                g.append(n)
                break
        else:
            groups.append([n])
    rest = [g[0] for g in groups]
    others = [n for n in rest if n not in (head, tail)]
    # the spine: tail end (or the point between the rearmost feet) to the head end
    if tail is not None:
        spine = nx.shortest_path(T, tail, head, weight='w')
    else:
        rear = sorted(feet, key=lambda n: P[n] @ fwd)[:2] or [head]
        paths = [nx.shortest_path(T, f, head, weight='w') for f in rear]
        common = [n for n in paths[0] if all(n in p for p in paths[1:])]
        spine = common if len(common) > 2 else paths[0]
    on_spine = {n: i for i, n in enumerate(spine)}

    def attach(end):
        path = nx.shortest_path(T, end, spine[0], weight='w')
        for k, n in enumerate(path):
            if n in on_spine:
                return path[:k + 1][::-1], on_spine[n]       # from the spine out to the end
        return path[::-1], 0
    limbs = []
    for n in feet + others:
        path, at = attach(n)
        if len(path) < 3:
            continue
        side = 1 if lat(P[n]) > 0 else -1
        length = float(np.sum(np.linalg.norm(np.diff(path_points(T, path), axis=0), axis=1)))
        limbs.append(dict(end=n, path=path, at=at, side=side, foot=n in feet, length=length))
    # one limb per stretch of path leaving the trunk: the toes of one foot, the claws of one hand, the fingers of one
    # wing share it (each claw on the floor counted as a foot gave a biped three leg pairs; a wing's membrane spikes
    # gave seven arm pairs); the limb ends at its furthest tip
    def key(limb):
        acc, prev = 0.0, limb['path'][0]
        for n in limb['path'][1:]:
            acc += T.edges[prev, n]['w'] if T.has_edge(prev, n) else 0.0
            prev = n
            if acc >= .12 * height:
                return n
        return limb['path'][-1]
    best = {}
    for limb in limbs:
        k = key(limb)
        if k not in best or limb['length'] > best[k]['length']:
            best[k] = limb
        if limb['foot']:
            best[k]['foot'] = True
    limbs = list(best.values())
    # pairs: mirrored ends, same kind, nearest
    pairs, used = [], set()
    for i, a in enumerate(limbs):
        if i in used:
            continue
        best, bd = None, 1e9
        for j, b in enumerate(limbs):
            if j == i or j in used or a['side'] == b['side'] or a['foot'] != b['foot']:
                continue
            pa, pb = P[a['end']], P[b['end']]
            d = np.linalg.norm((pa - (pa @ left) * left) - (pb - (pb @ left) * left))
            if d < bd:
                best, bd = j, d
        if best is not None and bd < .35 * height:
            used |= {i, best}
            pairs.append((a, limbs[best]))
    singles = [l for k, l in enumerate(limbs) if k not in used]
    # an unpaired end off the middle of a mirror-symmetric body is not a body part: a prop or a hilt touching the body
    # (the centaur's sword gave a chain from the chest), no bones
    props = [l for l in singles if not l['foot'] and abs(lat(P[l['end']])) > .1 * height]
    if props:
        log(f'autorig: {len(props)} unpaired end(s) off the middle: props, no bones')
    singles = [l for l in singles if l not in props]
    if singles:
        log(f'autorig: {len(singles)} unpaired limb end(s): a plain chain of bones each')
    leg_pairs = sorted([p for p in pairs if p[0]['foot']], key=lambda p: p[0]['at'])      # rear first
    # the main upper pair (arms, wings) first: the longest; a wyvern's wing claws and horns come after it
    up_pairs = sorted([p for p in pairs if not p[0]['foot']], key=lambda p: -p[0]['length'])
    if body == 'auto':
        body = ('centaur' if len(leg_pairs) >= 2 and up_pairs else 'quadruped' if len(leg_pairs) >= 2 else
                'winged' if up_pairs and np.ptp(path_points(T, up_pairs[0][0]['path']) @ left) > .5 * height
                else 'biped')
    log(f'autorig: body {body}: {len(leg_pairs)} leg pairs, {len(up_pairs)} upper pairs, tail {tail is not None}')
    return dict(fwd=fwd, up=up, left=left, spine=spine, head=head, tail=tail, leg_pairs=leg_pairs,
                up_pairs=up_pairs, body=body, floor=floor, singles=singles)


def build(T, A, height, voxel=.01, log=print):
    """Bones (name, head, tail, parent, role) from the analysed tree."""
    step = .01 * height
    win = max(2, int(round(.035 * height / step)))
    spine = smooth(resample(path_points(T, A['spine']), step)[0], 2)
    sp_s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(spine, axis=0), axis=1))])
    raw_s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path_points(T, A['spine']), axis=0), axis=1))])

    def spine_at(k):                        # a spine node index -> arc length on the resampled spine
        return raw_s[k] * sp_s[-1] / max(raw_s[-1], 1e-9)

    def point_at(s):
        s = np.clip(s, 0, sp_s[-1])
        return np.array([np.interp(s, sp_s, spine[:, k]) for k in range(3)])
    bones = []

    def add(name, head, tail, parent, role):
        bones.append(dict(name=name, head=[float(v) for v in head], tail=[float(v) for v in tail], parent=parent,
                          role=role))
        return name
    has_tail = A['tail'] is not None
    legs = A['leg_pairs']
    ups = A['up_pairs']
    body = A['body']
    hips_s = spine_at(legs[0][0]['at']) if legs else 0.0
    marks = sorted({spine_at(p[0]['at']) for p in legs + ups})
    last_s = max(marks) if marks else sp_s[-1] * .6
    root = add('Root', [point_at(hips_s)[0], point_at(hips_s)[1], A['floor']],
               [point_at(hips_s)[0], point_at(hips_s)[1], A['floor'] + .1 * height], None, 'root')
    # tail: from the hips back to the tail's end, three bones (hips side first)
    hips_head = point_at(hips_s)
    if has_tail and hips_s > .08 * height:
        # where the tail leaves the trunk: along the spine from the hips back, the body's radius falls below half
        rad = np.array([T.nodes[n]['r'] for n in A['spine']])
        k_hips = int(np.argmin(np.abs(raw_s - hips_s * raw_s[-1] / max(sp_s[-1], 1e-9))))
        base_k = next((k for k in range(k_hips, -1, -1) if rad[k] < .5 * rad[k_hips]), 0)
        base_s = min(spine_at(base_k), hips_s - .02 * height)
        # a long body behind the only legs (a belly slider, a seal, a slug-bodied boss: the trunk runs on more than
        # a body height behind the hips): the hips bone stays short and the whole back body is the tail chain, one
        # bone per ~.16 body heights (up to 10), so it can bend in a wave; other bodies keep their three tail bones
        seg = .16 * height
        long_back = hips_s - base_s > 1.0 * height
        if long_back:
            base_s = hips_s - 1.5 * seg
        tail_base_s = base_s
        hips = add('Hips', hips_head, point_at(base_s), root, 'trunk')
        n_tail = int(np.clip(round(base_s / seg), 3, 10)) if long_back else 3
        cuts = np.linspace(base_s, 0, n_tail + 1)
        parent = hips
        for k in range(n_tail):
            parent = add(f'Tail{k + 1}', point_at(cuts[k]), point_at(cuts[k + 1]), parent, 'tail')
    else:
        tail_base_s = hips_s
        hips = add('Hips', hips_head, point_at(hips_s + .1 * height), root, 'trunk')
    # trunk: hips -> (each limb pair's attachment) -> chest; three bones per long stretch
    stops = [m for m in marks if m > hips_s + .04 * height]
    # as many as the trunk needs: a robed figure's ran from the hem to the shoulders and wanted a ninth
    names = (f'Spine{k}' for k in range(1, 1000))
    parent, s0 = hips, hips_s
    attach_bone = {round(hips_s, 6): hips}
    for m in stops:
        n_seg = max(1, int(round((m - s0) / (.12 * height))))
        for c in np.linspace(s0, m, n_seg + 1)[1:]:
            parent = add(next(names), point_at(s0), point_at(c), parent, 'trunk')
            s0 = c
        attach_bone[round(m, 6)] = parent
    # neck and head: from the last attachment to the head end
    rest = sp_s[-1] - s0
    head_base = s0 + .55 * rest
    neck_cuts = np.linspace(s0, head_base, 3)
    for k in range(2):
        parent = add(f'Neck{k + 1}', point_at(neck_cuts[k]), point_at(neck_cuts[k + 1]), parent, 'neck')
    add('Head', point_at(head_base), point_at(sp_s[-1]), parent, 'head')

    def bone_for(at_s):
        k = min(attach_bone, key=lambda m: abs(m - at_s))
        return attach_bone[k]
    # limbs
    kinds = []
    if body in ('quadruped', 'centaur'):
        kinds += [('hind', legs[0])] + ([('front', legs[1])] if len(legs) > 1 else [])
        kinds += [('arm', ups[0])] if ups else []
    elif body == 'winged':
        kinds += [('biped', legs[0])] if legs else []
        kinds += [('wing', ups[0])] if ups else []
    else:
        kinds += [('biped', legs[0])] if legs else []
        kinds += [('arm', ups[0])] if ups else []
    # whatever the body plan does not name (a third pair of legs, a lone arm, a tentacle): a plain chain of bones
    # along its centre line, one about every 12% of the height; the reader and the fit need joints, not names
    known = {id(p) for _, p in kinds}
    extra = [(l, f'Limb{k + 1}') for k, l in enumerate(A['singles'])]
    for k, p in enumerate([p for p in legs + ups if id(p) not in known]):
        extra += [(l, f'Pair{k + 1}') for l in p]
    for limb, base in extra:
        sfx = SIDES[limb['side']] if limb in [x for p in legs + ups for x in p] else ''
        Pl = smooth(resample(path_points(T, limb['path']), step)[0], 1.5)
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pl, axis=0), axis=1))])
        n = max(2, int(round(L[-1] / (.12 * height))))
        cuts = [Pl[int(np.argmin(np.abs(L - f * L[-1])))] for f in np.linspace(0, 1, n + 1)]
        parent = bone_for(spine_at(limb['at']))
        for i in range(n):
            parent = add(f'{base}_{i + 1}{sfx}', cuts[i], cuts[i + 1], parent, 'chain')
    for kind, pair in kinds:
        top, names_k, fracs = LEG_PLANS[kind]
        paths = []
        for limb in pair:
            Pl, rl = limb_path(T, limb['path'], step)
            r0 = T.nodes[limb['path'][0]]['r']                     # the trunk's radius where the limb meets it
            e = exit_index(Pl, spine, r0)
            Pe, re = (Pl[e:], rl[e:]) if len(Pl) - e >= 4 else (Pl, rl)
            if kind == 'arm':
                Pe, re = cut_held_prop(Pe, re, height, voxel * height, log, 'arm' + SIDES[limb['side']])
            paths.append(Pe)
        if kind == 'arm' and len(paths) == 2:
            lengths = [float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))) for q in paths]
            if min(lengths) < .85 * max(lengths):
                k = int(np.argmin(lengths))
                log(f'autorig: arm{SIDES[pair[k]["side"]]} {lengths[k]:.2f} x height, its twin {max(lengths):.2f}: '
                    f'made as long')
                paths[k] = extend_to(paths[k], max(lengths), height)
        for limb, Pe in zip(pair, paths):
            sfx = SIDES[limb['side']]
            r0 = T.nodes[limb['path'][0]]['r']
            joints = [Pe[0]] + place(Pe, fracs, win) + [Pe[-1]]
            if kind == 'hind' and has_tail:
                # a quadruped's hip socket sits at the back of the trunk, near the tail's root, above the knee's line;
                # the centre line leaves the trunk well in front of it (a wolf: 8 cm)
                back = point_at(tail_base_s + .12 * (hips_s - tail_base_s + .25 * height))
                joints[0] = np.array([joints[1] @ A['left'] * A['left'][0] * .9 + back[0] * (1 - abs(A['left'][0])),
                                      back[1] if abs(A['left'][0]) > .5 else joints[1][1] * .9,
                                      back[2] - .25 * r0])
                if abs(A['left'][0]) > .5:
                    joints[0][0] = (joints[1] @ A['left']) * .9 * A['left'][0]
            parent = bone_for(spine_at(limb['at']))
            if top is not None:
                # a scapula from the top of the back (a quadruped's withers) or a collarbone from the chest's middle
                base = Pl[0] + (A['up'] * .7 * r0 - A['fwd'] * .25 * r0 if kind == 'front' else 0)
                parent = add(top + sfx, base, joints[0], parent, kind)
            for k, nm in enumerate(names_k):
                parent = add(nm + sfx, joints[k], joints[k + 1], parent, kind)
    return bones


def sheet(T, bones, path, height):
    import cv2
    pts = np.array([T.nodes[n]['pos'] for n in T.nodes])
    views = [(1, 2, 'side (Y, Z)'), (0, 2, 'front (X, Z)'), (0, 1, 'top (X, Y)')]
    S = 360
    tiles = []
    lo, hi = pts.min(0), pts.max(0)
    span = (hi - lo).max() * 1.1
    for a, b, label in views:
        img = np.full((S, S, 3), 245, np.uint8)
        to = lambda p: (int((p[a] - (lo[a] + hi[a]) / 2) / span * S + S / 2), int(S / 2 - (p[b] - (lo[b] + hi[b]) / 2) / span * S))
        for p in pts[::2]:
            cv2.circle(img, to(p), 1, (190, 190, 190), -1)
        for bn in bones:
            c = (40, 140, 255) if bn['name'].endswith('.L') else (255, 120, 40) if bn['name'].endswith('.R') else (40, 160, 40)
            cv2.line(img, to(bn['head']), to(bn['tail']), c, 2, cv2.LINE_AA)
            cv2.circle(img, to(bn['head']), 3, (20, 20, 20), -1)
        cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        tiles.append(img)
    cv2.imwrite(str(path), np.concatenate(tiles, 1))


def main():
    ap = argparse.ArgumentParser(prog='video2rig.autorig.skeleton')
    ap.add_argument('work')
    ap.add_argument('--body', default='auto')
    ap.add_argument('--voxel', type=float, default=.01, help='voxel size, a share of the height')
    ap.add_argument('--facing', default='auto', help='auto (from the shape), -Y, +Y, -X or +X')
    a = ap.parse_args()
    work = Path(a.work)
    if a.body == 'auto':
        from . import manylegs
        if manylegs.build(work):
            from .unirig import sheet_only
            sheet_only(work)
            return
    d = np.load(work / 'shell.npz')
    height = float(d['height'])
    solid, lo = voxelize(d['verts'].astype(np.float64), d['faces'], a.voxel * height)
    T = centre_tree(solid, lo, a.voxel * height)
    T = prune(T, .05 * height)
    T = collapse_tips(T, .15 * height)
    T = prune(T, .05 * height)
    facing = a.facing if a.facing != 'auto' else find_facing(solid, lo, a.voxel * height, height)
    vox = lo + (np.argwhere(solid) + .5) * a.voxel * height
    mid = float(np.median(vox @ np.cross([0, 0, 1.], AXES[facing])))
    A = analyse(T, height, facing, a.body, mid=mid)
    bones = build(T, A, height, a.voxel)
    (work / 'joints.json').write_text(json.dumps({'body': A['body'], 'facing': facing, 'height': height,
                                                 'left': [float(v) for v in A['left']], 'bones': bones}, indent=1)
                                      + '\n', encoding='utf-8')
    sheet(T, bones, work / 'skeleton-sheet.png', height)
    print(f'AUTORIG_SKELETON {A["body"]}: {len(bones)} bones -> {work / "joints.json"}', flush=True)


if __name__ == '__main__':
    main()
