"""Which keypoints each training picture shows: the creature's own mesh in the pose, rasterized through the panel's
camera (the tool's z-buffer, raster.py), and a keypoint counts as seen when the surface around its pixel belongs to
its own limb segment (its bone, the bone above, the bones below). A far leg behind the near one, an elbow behind
the chest: hidden. The reader is taught to answer "nothing" there instead of a confident guess, so the fit takes
those joints from the other panel and the skeleton.

Needs only the poses (not the pictures), so it runs beside the Blender render:
    python -m video2rig.reader.visible <project> [--workers 6]       -> <project>/reader/visible.npz (vis (N, V, K))
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

WINDOW = 2          # px (at the raster's scale) round the keypoint where its own surface may show
SCALE = .5          # raster at half the training picture's size


def triangle_bones(cr):
    """The bone that carries most of each triangle's skin weight."""
    nb = cr.rig.nb
    W = np.zeros((len(cr.idx), int(cr.idx.max()) + 1))
    np.add.at(W, (np.arange(len(cr.idx))[:, None], cr.idx), cr.wts)
    tw = W[cr.mesh['tris']].sum(1)
    return np.argmax(tw, axis=1)                                    # a prop's triangles: nb + its index


def neighbours(rig, j):
    out = {j}
    if rig.parents[j] >= 0:
        out.add(int(rig.parents[j]))
    out |= {k for k in range(rig.nb) if rig.parents[k] == j}
    return out


def _work(job):
    root, lo, hi, poses = job
    os.environ['V2R_CPU'] = '1'                              # the card renders meanwhile
    from ..common import Project
    from ..creature import Creature
    from ..raster import rasterize
    from . import KeypointRig, keypoints, project_ortho, tile_sizes
    from .net import SIZE
    pr = Project(root)
    cfg = pr.clip_cfg('')
    cfg.setdefault('solve', {})
    cr = Creature(pr, cfg)
    rig = cr.rig
    kps = keypoints(cr)
    kr = KeypointRig(cr, kps)
    params = np.load(poses)['params'][lo:hi]
    cams = json.loads((pr.start_dir / 'cameras.json').read_text(encoding='utf-8'))
    M = np.array(cr.data['armature_world'], float)
    sizes = [(max(1, int(round(w * SCALE))), max(1, int(round(h * SCALE)))) for w, h in tile_sizes(cams, SIZE)]
    tri_bone = triangle_bones(cr)
    allow = np.zeros((len(kps), rig.nb + len(rig.props)), bool)
    prop_names = [pr['name'] for pr in rig.props]
    for k, kp in enumerate(kps):
        if str(kp[1]).startswith('@'):
            allow[k, rig.nb + prop_names.index(kp[1][1:])] = True
        else:
            allow[k, list(neighbours(rig, rig.names.index(kp[1])))] = True
    tris = cr.mesh['tris']
    vis = np.zeros((len(params), len(cams['views']), len(kps)), bool)
    for n, p in enumerate(params):
        X = rig.skin(p, cr.mesh['coords'], cr.idx, cr.wts)
        Xk = kr.points(p)
        for v, rec in enumerate(cams['views']):
            size = sizes[v]
            pix = project_ortho(rec, M, X, size)
            depth = (X @ M[:3, :3].T + M[:3, 3] - np.asarray(rec['location'])) @ np.asarray(rec['forward'])
            tri, _, _ = rasterize(pix, depth, tris, size)
            bone = np.where(tri >= 0, tri_bone[np.maximum(tri, 0)], -1)
            uv = project_ortho(rec, M, Xk, size)
            for k in range(len(kps)):
                x, y = int(uv[k, 0]), int(uv[k, 1])
                win = bone[max(0, y - WINDOW):y + WINDOW + 1, max(0, x - WINDOW):x + WINDOW + 1]
                win = win[win >= 0]
                vis[n, v, k] = bool(len(win)) and bool(allow[k, win].any())
    return lo, vis


def compute(project_root, workers=6, log=print, poses=None, out=None):
    root = str(Path(project_root).resolve())
    poses = str(Path(poses).resolve()) if poses else str(Path(root) / 'reader' / 'poses.npz')
    out = Path(out) if out else Path(root) / 'reader' / 'visible.npz'
    n = len(np.load(poses)['params'])
    step = (n + workers - 1) // workers
    jobs = [(root, lo, min(n, lo + step), poses) for lo in range(0, n, step)]
    t0 = time.time()
    parts = {}
    with ProcessPoolExecutor(workers) as ex:
        for lo, vis in ex.map(_work, jobs):
            parts[lo] = vis
    vis = np.concatenate([parts[lo] for lo in sorted(parts)])
    np.savez_compressed(out, vis=vis)
    log(f'VISIBLE {n} poses in {time.time() - t0:.1f} s: seen in the panels {[round(float(vis[:, v].mean()), 2) for v in range(vis.shape[1])]}')
    return vis


def main():
    ap = argparse.ArgumentParser(prog='video2rig.reader.visible')
    ap.add_argument('project')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--poses', default='', help='the poses (default <project>/reader/poses.npz); a render set of its '
                    'own keeps its poses.npz and visible.npz inside its renders folder')
    ap.add_argument('--out', default='', help='default <project>/reader/visible.npz')
    a = ap.parse_args()
    compute(a.project, a.workers, poses=a.poses, out=a.out)


if __name__ == '__main__':
    main()
