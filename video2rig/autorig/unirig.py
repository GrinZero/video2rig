"""Auto-rig, step 2 with UniRig: a learned skeleton, checked and fixed by rules measured on the mesh.

    python -m video2rig.autorig.unirig <work> [<work> ...] [--draws 4] [--unirig DIR] [--blender EXE]
        -> <work>/joints.json (the same contract as video2rig.autorig.skeleton) and <work>/skeleton-sheet.png

UniRig (VAST-AI-Research, MIT, https://github.com/VAST-AI-Research/UniRig) draws a skeleton for any shape, with
joints where an artist puts them (elbows, knees, wrists, wing fingers). Its draws differ from run to run and it does
not know every body plan (a centaur: no front horse legs in five draws), so:
1. a few draws in one run; each is scored on what the mesh shows for sure: every place the model stands on the
   floor has a bone reaching it, and limbs mirror each other where the mesh does; the best draw is kept;
2. fingers, toes and other short ends at one joint fold into that joint's bone (a hand, a foot, a head): too small to
   read in a video, they move with it; branches lying mostly outside the skin are dropped, bone ends poking out of it
   pulled back in;
3. a floor contact no bone reaches gets the leg of our own centre-line skeleton (video2rig.autorig.skeleton);
4. the bones get the names the tool reads, from the same body analysis as the centre-line skeleton (feet on the
   floor, head in front, tail behind, mirrored pairs): joints keep UniRig's places, a name goes to the joint nearest
   to where a body of that kind has it (a knee about a third down a horse's hind leg);
5. a standing leg's last bone runs along its sole to the toe tips, and a head UniRig ended in an ear goes from the
   skull to the front of the face;
6. where the mesh is mirror-symmetric around a limb, its twin is made its exact mirror (both moved to their mean),
   and middle bones sit on the middle plane.
Without UniRig (not found) the centre-line skeleton is used.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import manylegs
from . import skeleton as S

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = r'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe'
TASK = 'configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml'
LIMB_KINDS = ('hind', 'front', 'biped', 'arm', 'wing')
DIGIT = .22          # an end shorter than this (x height) from its fork is a finger, a claw, a toe, an ear: folded
# the names along a limb after its optional top bone, and where (share of the limb's length from its root joint)
# each next joint sits on a body of that kind: the same plans as the centre-line skeleton
PLANS = {kind: (top, names, fr) for kind, (top, names, fr) in S.LEG_PLANS.items()}


# ----- finding the programs -----------------------------------------------------------------------------------
def find_blender(given):
    for c in (given, os.environ.get('V2R_BLENDER'), DEFAULT_BLENDER, shutil.which('blender')):
        if c and Path(c).exists():
            return str(c)
    raise SystemExit('Blender not found: pass --blender or set V2R_BLENDER')


def find_unirig(given):
    for c in (given, os.environ.get('V2R_UNIRIG'), REPO.parent / 'unirig', Path.home() / 'unirig'):
        if c and (Path(c) / 'run.py').exists():
            return Path(c)
    return None


def unirig_python(d):
    for p in (d / '.venv' / 'Scripts' / 'python.exe', d / '.venv' / 'bin' / 'python'):
        if p.exists():
            return str(p)
    return sys.executable


def blender_io(blender, args, blend=None, py=None):
    script = REPO / 'blender' / 'unirig_io.py'
    cmd = [blender, '-b', '--factory-startup'] + ([str(blend)] if blend else []) + [
        '--python', str(script), '--'] + [str(a) for a in args]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode('utf-8', 'replace')
    if (r.returncode or 'UNIRIG_IO_OK' not in out) and py and not blend:
        # Blender 5.1's FBX importer dies without a word (heap corruption) on some skeletons UniRig drew (the Leshy,
        # 2026-10-20) that UniRig's own bpy wrote and reads: the same script there
        code = ("import sys; sys.argv = ['unirig_io.py', '--'] + sys.argv[1:]; "
                f"exec(compile(open(r'{script}', encoding='utf-8').read(), 'unirig_io.py', 'exec'), "
                "{'__name__': '__main__'})")
        r2 = subprocess.run([py, '-c', code] + [str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out2 = r2.stdout.decode('utf-8', 'replace')
        if not r2.returncode and 'UNIRIG_IO_OK' in out2:
            print(f'autorig: Blender ({Path(blender).parent.name}) could not read the skeletons (exit {r.returncode}): '
                  'read with UniRig\'s own bpy', flush=True)
            return
        out += '\n--- with UniRig\'s bpy:\n' + out2
    if r.returncode or 'UNIRIG_IO_OK' not in out:
        raise RuntimeError('unirig_io.py failed:\n' + out[-3000:])


def draw_all(works, draws, seed, unirig, blender, log):
    """One UniRig run for every work folder: `draws` copies of each model, each copy a different draw.
    Returns {work: [skeleton.fbx, ...]}."""
    import time
    run = Path(works[0]) / 'unirig'
    if run.exists():
        shutil.rmtree(run)
    for d in ('src', 'in', 'npz'):
        (run / d).mkdir(parents=True)
    for i, w in enumerate(works):
        glb = Path(w) / 'model.glb'
        if not glb.exists():
            blender_io(blender, ['export', glb], blend=Path(w) / 'model.blend')
        shutil.copy(glb, run / 'src' / f'm{i}.glb')
    py = unirig_python(unirig)
    log(f'autorig: UniRig reads {len(works)} model(s) ...')
    with open(run / 'extract.log', 'wb') as f:
        subprocess.run([py, '-m', 'src.data.extract', '--config=configs/data/quick_inference.yaml',
                        '--require_suffix=obj,fbx,FBX,dae,glb,gltf,vrm', '--force_override=true', '--num_runs=1',
                        '--id=0', f'--time={time.strftime("%Y_%m_%d_%H_%M_%S")}', '--faces_target_count=50000',
                        f'--input_dir={run / "src"}', f'--output_dir={run / "npz"}'],
                       cwd=unirig, stdout=f, stderr=subprocess.STDOUT)
    for i, w in enumerate(works):
        got = run / 'npz' / f'm{i}' / 'raw_data.npz'
        if not got.exists():
            log(f'autorig: UniRig could not read {w} (see {run / "extract.log"})')
            continue
        for k in range(draws):          # the same mesh under another name: one more draw
            shutil.copytree(got.parent, run / 'npz' / f'm{i}_draw{k}')
            shutil.copy(run / 'src' / f'm{i}.glb', run / 'in' / f'm{i}_draw{k}.glb')
    log(f'autorig: UniRig draws {draws} skeleton(s) per model ...')
    with open(run / 'unirig.log', 'wb') as f:
        subprocess.run([py, 'run.py', f'--task={TASK}', f'--seed={seed}', f'--input_dir={run / "in"}',
                        f'--output_dir={run / "out"}', f'--npz_dir={run / "npz"}'],
                       cwd=unirig, stdout=f, stderr=subprocess.STDOUT)
    res = {}
    for i, w in enumerate(works):
        res[w] = [p for k in range(draws) for p in [run / 'out' / f'm{i}_draw{k}' / 'skeleton.fbx'] if p.exists()]
    return res


# ----- the mesh -------------------------------------------------------------------------------------------------
class Mesh:
    """The model's watertight shell (autorig_prepare.py): voxels for inside tests, its middle plane, floor contacts."""

    def __init__(self, work, facing='auto'):
        d = np.load(Path(work) / 'shell.npz')
        self.verts, faces = d['verts'].astype(np.float64), d['faces']
        if 'skin_verts' in d.files:
            # with the model's own skin: one-sided sheets the remesh lost (a robe, a hood) are body too
            faces = np.concatenate([faces, d['skin_faces'] + len(self.verts)])
            self.verts = np.concatenate([self.verts, d['skin_verts'].astype(np.float64)])
        self.height = h = float(d['height'])
        self.lo_m, self.hi_m = np.asarray(d['lo'], float), np.asarray(d['hi'], float)
        self.size = .01 * h
        self.solid, self.lo = S.voxelize(self.verts, faces, self.size)
        # from the shape unless given: a big shell on the back (a bell) put the top of a hunched giant behind it
        self.facing = facing if facing in S.AXES else S.find_facing(self.solid, self.lo, self.size, h,
                                                                     log=lambda m: None)
        ijk = np.argwhere(self.solid)
        self.vox = self.lo + (ijk + .5) * self.size
        # how deep inside the body each voxel lies (in voxels): 1 on a thin sheet, more in flesh
        self.depth = ndimage.distance_transform_edt(self.solid)[tuple(ijk.T)]
        self.fwd = S.AXES[self.facing]
        self.left = np.cross([0, 0, 1.], self.fwd)
        self.mid = float(np.median(self.vox @ self.left))
        self.kd = cKDTree(self.vox)
        self.floor = float(self.verts[:, 2].min())
        self.patches = self.floor_patches()

    def inside(self, P):
        P = np.atleast_2d(P)
        ijk = np.floor((P - self.lo) / self.size).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.solid.shape)), axis=1)
        res = np.zeros(len(P), bool)
        res[ok] = self.solid[tuple(ijk[ok].T)]
        return res

    def off_skin(self, P):
        """How far each point lies from the body (0 inside; a thin membrane counts as body)."""
        d, _ = self.kd.query(np.atleast_2d(P))
        return np.where(self.inside(P), 0.0, d)

    def mirror(self, P):
        P = np.asarray(P, float)
        return P - 2 * ((P @ self.left) - self.mid)[..., None] * self.left

    def symmetric_near(self, P, radius=.05):
        """Share of the solid voxels around the points whose mirror image is solid too (1 = mirror-symmetric)."""
        idx = sorted({i for l in self.kd.query_ball_point(np.atleast_2d(P), radius * self.height) for i in l})
        if not idx:
            return 0.0
        return float(self.inside(self.mirror(self.vox[idx])).mean())

    def flatness(self, limb_pts, body_pts):
        """How flat the body is across a limb (width over thickness of its cross-section, the median of four cuts at
        30-75% of its length; only what lies nearer to the limb than to the trunk): a wing about 10, an arm 1-2.5
        however long or thick (a giant's arms reach its knees, a T-posed man spreads his as wide as a bat)."""
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(limb_pts, axis=0), axis=1))])
        kd_l, kd_b = cKDTree(limb_pts), cKDTree(body_pts)
        out = []
        for t in (.3, .45, .6, .75):
            i = int(np.searchsorted(s, t * s[-1]))
            p = limb_pts[i]
            ax = limb_pts[min(i + 3, len(limb_pts) - 1)] - limb_pts[max(i - 3, 0)]
            ax = ax / max(np.linalg.norm(ax), 1e-12)
            V = self.vox[np.array(self.kd.query_ball_point(p, .35 * self.height), int)]
            V = V[np.abs((V - p) @ ax) < 1.5 * self.size]
            if len(V):
                V = V[kd_l.query(V)[0] < kd_b.query(V)[0]]
            if len(V) < 5:
                continue
            Q = (V - p) - np.outer((V - p) @ ax, ax)
            w = np.sort(np.linalg.eigvalsh(np.cov(Q.T)))[::-1]
            out.append(float(np.sqrt(w[0] / max(w[1], 1e-12))))
        return float(np.median(out)) if out else 1.0

    def floor_patches(self):
        """Where the model stands: vertices within 2% of the height above the floor, in patches (8% apart = one)."""
        h = self.height
        P = self.verts[self.verts[:, 2] < self.floor + .02 * h][:, :2]
        P = P[::max(1, len(P) // 1500)]
        if not len(P):
            return []
        G = nx.Graph()
        G.add_nodes_from(range(len(P)))
        G.add_edges_from(cKDTree(P).query_pairs(.08 * h))
        patches = [P[sorted(c)] for c in nx.connected_components(G) if len(c) >= 4]
        # cloth that reaches the floor is no foot: where everything above the patch (up to 8% of the height, 3%
        # beside it) is a sheet one voxel thick, nothing stands there (the ragged ends of a robe's strips: a
        # hovering figure "stood" on two of them and was to get a leg for each)
        low = self.vox[:, 2] < self.floor + .08 * h
        if not low.any():
            return patches
        kd = cKDTree(self.vox[low][:, :2])
        deep = self.depth[low]
        kept = []
        for p in patches:
            near = sorted({i for l in kd.query_ball_point(p, .03 * h) for i in l})
            if not near or deep[near].max() >= 2.0:
                kept.append(p)
        return kept

    def uncovered(self, segments):
        """Floor patches no bone reaches (a bone point within 15% of the height above the floor, 6% beside it)."""
        h = self.height
        pts = np.array([a + (b - a) * t for a, b in segments for t in np.linspace(0, 1, 11)])
        pts = pts[pts[:, 2] < self.floor + .15 * h][:, :2] if len(pts) else np.zeros((0, 2))
        if not len(pts):
            return list(range(len(self.patches)))
        kd = cKDTree(pts)
        return [k for k, p in enumerate(self.patches) if kd.query(p)[0].min() > .06 * h]


# ----- a UniRig draw as a joint tree ----------------------------------------------------------------------------
def to_model(draw, mesh, log):
    """UniRig's bones in the model's own frame (it drew them on a copy scaled into its own box)."""
    lo_n, hi_n = np.array(draw['lo']), np.array(draw['hi'])
    ext_m, ext_n = mesh.hi_m - mesh.lo_m, hi_n - lo_n
    s = ext_m.max() / max(ext_n.max(), 1e-9)
    if np.abs(ext_n * s - ext_m).max() > .05 * ext_m.max():
        log(f'autorig: warning, UniRig box {np.round(ext_n * s, 3)} vs model {np.round(ext_m, 3)}')
    c_m, c_n = (mesh.lo_m + mesh.hi_m) / 2, (lo_n + hi_n) / 2
    f = lambda p: (np.asarray(p, float) - c_n) * s + c_m
    return [dict(b, head=f(b['head']), tail=f(b['tail'])) for b in draw['bones']]


def joint_tree(bones):
    """Nodes: every bone's head (its index) and, for bones without children, their tail (index n + i, tip=True)."""
    n = len(bones)
    idx = {b['name']: i for i, b in enumerate(bones)}
    G = nx.Graph()
    for i, b in enumerate(bones):
        G.add_node(i, pos=np.asarray(b['head'], float), tip=False, r=0.0)
    for i, b in enumerate(bones):
        if b['parent'] in idx:
            p = idx[b['parent']]
            G.add_edge(p, i, w=float(np.linalg.norm(G.nodes[p]['pos'] - G.nodes[i]['pos'])) + 1e-6)
    has_kid = {idx[b['parent']] for b in bones if b['parent'] in idx}
    for i, b in enumerate(bones):
        if i not in has_kid:
            G.add_node(n + i, pos=np.asarray(b['tail'], float), tip=True, r=0.0)
            G.add_edge(i, n + i, w=float(np.linalg.norm(G.nodes[i]['pos'] - G.nodes[n + i]['pos'])) + 1e-6)
    root = next((i for i, b in enumerate(bones) if b['parent'] not in idx), 0)
    G = G.subgraph(nx.node_connected_component(G, root)).copy()
    return G, root


def add_tip(G, j, p):
    t = max(G.nodes) + 1
    G.add_node(t, pos=np.asarray(p, float), tip=True, r=0.0)
    G.add_edge(j, t, w=float(np.linalg.norm(G.nodes[j]['pos'] - p)) + 1e-6)


def fold_digits(G, root, height):
    """Where every branch leaving a joint ends within DIGIT x height (fingers at a wrist, toes at an ankle, ears and a
    jaw at a head), the branches go and the joint's bone reaches to the mean of their furthest ends."""
    folded = 0
    while True:
        par = {c: q for q, c in nx.bfs_edges(G, root)}
        kids = {}
        for c, p in par.items():
            kids.setdefault(p, []).append(c)
        done = False
        for f, ch in kids.items():
            if len(ch) < 2 or G.nodes[f]['tip']:
                continue
            subs = []
            for c in ch:
                H = G.copy()
                H.remove_edge(f, c)
                comp = nx.node_connected_component(H, c)
                d = nx.single_source_dijkstra_path_length(G.subgraph(comp | {f}), f, weight='w')
                far = max(comp, key=lambda n: d[n])
                subs.append((comp, d[far], G.nodes[far]['pos']))
            if all(s[1] < DIGIT * height for s in subs):
                end = np.mean([s[2] for s in subs], axis=0)
                for comp, _, _ in subs:
                    G.remove_nodes_from(comp)
                add_tip(G, f, end)
                folded += len(subs)
                done = True
                break
        if not done:
            return folded


def drop_outside(G, root, mesh):
    """Branches lying mostly outside the skin go (a stray bone out of a centaur's back); end points outside are pulled
    back along their bone until inside (at most to 30% of it)."""
    dropped = 0
    tol = 2.5 * mesh.size
    for t in [n for n in G.nodes if G.nodes[n]['tip']]:
        if t not in G:
            continue
        path, n = [t], t
        while True:
            nb = [m for m in G.neighbors(n) if m not in path]
            if not nb:
                break
            n = nb[0]
            path.append(n)
            if G.degree(n) != 2 or n == root:
                break
        if root in path[:-1] or len(path) < 2:
            continue
        pts = np.array([G.nodes[a]['pos'] + (G.nodes[b]['pos'] - G.nodes[a]['pos']) * s
                        for a, b in zip(path, path[1:]) for s in np.linspace(0, 1, 6)])
        if (mesh.off_skin(pts) > tol).mean() > .5:
            G.remove_nodes_from(path[:-1])
            dropped += 1
    for t in [n for n in G.nodes if G.nodes[n]['tip']]:
        j = next(iter(G.neighbors(t)))
        a, b = G.nodes[j]['pos'], G.nodes[t]['pos'].copy()
        for s in np.linspace(1, .3, 15):
            if mesh.off_skin(a + (b - a) * s)[0] <= 1.5 * mesh.size:
                b = a + (b - a) * s
                break
        else:
            b = a + (b - a) * .3
        b[2] = max(b[2], mesh.floor)
        G.nodes[t]['pos'] = b
    return dropped


def clean(G, root, mesh):
    # a root bone with a single child (UniRig's pelvis stub) is no end of the body
    while not G.nodes[root]['tip'] and G.degree(root) == 1:
        nxt = next(iter(G.neighbors(root)))
        G.remove_node(root)
        root = nxt
    f = fold_digits(G, root, mesh.height)
    d = drop_outside(G, root, mesh)
    if root not in G:
        root = max(G.nodes, key=G.degree)
    G = G.subgraph(nx.node_connected_component(G, root)).copy()
    # a branch left without its tip (its end dropped) ends at its last joint
    for n in list(G.nodes):
        if not G.nodes[n]['tip'] and G.degree(n) == 1 and n != root:
            p = next(iter(G.neighbors(n)))
            a, b = G.nodes[p]['pos'], G.nodes[n]['pos']
            add_tip(G, n, b + (b - a) * .3)
    return G, root, f, d


def segments(G):
    return [(G.nodes[a]['pos'], G.nodes[b]['pos']) for a, b in G.edges]


def mirror_gap(G, mesh):
    """90th percentile over joints of the distance from a joint's mirror image to the nearest joint (x height)."""
    P = np.array([G.nodes[n]['pos'] for n in G.nodes])
    d, _ = cKDTree(P).query(mesh.mirror(P))
    return float(np.percentile(d, 90)) / mesh.height


# ----- names --------------------------------------------------------------------------------------------------
def choose(f, q, skip=.25):
    """Increasing indices into the joint fractions f for as many of the planned fractions q as fit, the total distance
    smallest (a planned joint left out costs `skip`). Returns {plan index: joint index}."""
    M, K = len(f), len(q)
    INF = 1e9
    D = np.full((K + 1, M + 1), INF)
    D[0, :] = 0.0
    back = {}
    for k in range(1, K + 1):
        for m in range(0, M + 1):
            best = (D[k - 1, m] + skip, ('skip', m))
            if m >= 1:
                c = D[k - 1, m - 1] + abs(f[m - 1] - q[k - 1])
                if c < best[0]:
                    best = (c, ('take', m - 1))
            if m >= 1 and D[k, m - 1] < best[0]:
                best = (D[k, m - 1], ('pass', m - 1))
            D[k, m] = best[0]
            back[k, m] = best[1]
    out, k, m = {}, K, M
    while k > 0:
        how, mm = back[k, m]
        if how == 'take':
            out[k - 1] = mm
            k, m = k - 1, mm
        elif how == 'skip':
            k -= 1
        else:
            m = mm
    return out


def label(G, hips, A, mesh, kinds, log):
    """Bones (name, head, tail, parent, role) on the joints of the cleaned UniRig tree."""
    h = mesh.height
    par = {c: q for q, c in nx.bfs_edges(G, hips)}
    kids = {}
    for c, p in par.items():
        kids.setdefault(p, []).append(c)
    pos = lambda n: G.nodes[n]['pos']
    joint = lambda n: not G.nodes[n]['tip']
    name, role, tail = {}, {}, {}
    spine = A['spine']
    hk = spine.index(hips)
    # the spine: tail part behind the hips (towards spine[0]), the front part to the head end
    front = [n for n in spine[hk + 1:] if joint(n)]
    back = [n for n in spine[:hk][::-1] if joint(n)]
    attach = [p[0]['at'] for p in A['leg_pairs'] + A['up_pairs'][:1]]      # legs and the arms/wings: the trunk
    last = max([a for a in attach if a > hk], default=hk)
    name[hips], role[hips] = 'Hips', 'trunk'
    if back:
        tail[hips] = pos(back[0])
    elif hk + 1 < len(spine):
        tail[hips] = pos(spine[hk + 1])
    for k, n in enumerate(back):
        name[n], role[n] = f'Tail{k + 1}', 'tail'
        i = spine.index(n)
        tail[n] = pos(spine[i - 1]) if i > 0 else None
    head_j = front[-1] if front else None
    ns = nn = 0
    for n in front:
        i = spine.index(n)
        tail[n] = pos(spine[i + 1]) if i + 1 < len(spine) else None
        if n == head_j:
            name[n], role[n] = 'Head', 'head'
        elif i <= last:
            ns += 1
            name[n], role[n] = f'Spine{ns}', 'trunk'
        else:
            nn += 1
            name[n], role[n] = f'Neck{nn}', 'neck'
    # limbs
    named_limbs = []
    for kind, pair in kinds:
        for limb in pair:
            named_limbs.append((kind, limb))
    gone, ends_at = set(), {}
    for kind, limb in named_limbs:
        sfx = S.SIDES[limb['side']] if kind != 'single' else ''
        path = [n for n in limb['path'][1:]]
        # a spur beside the limb is no hand: where the limb forks into its own continuation, which has a joint of
        # its own further on (a wrist), and single bones that end nearer than the limb does, the single bones go
        # and the limb runs on. UniRig drew a wraith's bare forearm as two bones side by side (radius and ulna);
        # the elbow then counted as the hand and the wrist was lost. Arms only: a leg's spur is a heel or a
        # dewclaw, its fork is the foot.
        if kind == 'arm':
            for k, n in enumerate(path[:-1]):
                ch = kids.get(n, []) if joint(n) else []
                if len(ch) < 2 or path[k + 1] not in ch:
                    continue
                spurs = [c for c in ch if c != path[k + 1]]
                if not joint(path[k + 1]) or any(c in gone for c in spurs):
                    break
                reach = np.linalg.norm(pos(path[-1]) - pos(n))

                def lone(c):                                # one bone, ending nearer than the limb's own end
                    sub = [c] + [m for m in nx.descendants(nx.bfs_tree(G, hips), c)]
                    return sum(1 for m in sub if joint(m)) <= 1 and \
                        max(np.linalg.norm(pos(m) - pos(n)) for m in sub) <= .85 * reach
                if not all(lone(c) for c in spurs):
                    break
                for c in spurs:
                    gone |= {c} | set(nx.descendants(nx.bfs_tree(G, hips), c))
                kids[n] = [path[k + 1]]
        # up to the first joint where it forks: a wing's rays beyond it stay bones named after it; an arm's or a leg's
        # fingers, claws and toes (a giant's claws reach a quarter of its height) fold into the hand or foot, its
        # bone reaching to the mean of their ends
        cut = next((k for k, n in enumerate(path) if joint(n) and len([c for c in kids.get(n, [])]) > 1), None)
        if cut is not None:
            path = path[:cut + 1]
            if kind in ('arm', 'biped', 'hind', 'front'):
                fork = path[-1]
                sub = set(nx.descendants(nx.bfs_tree(G, hips), fork))
                ends = []
                for c in kids[fork]:
                    tree_c = {c} | set(nx.descendants(nx.bfs_tree(G, hips), c))
                    ends.append(max(tree_c, key=lambda n: np.linalg.norm(pos(n) - pos(fork))))
                gone |= sub
                path = path + ['__end__']
                G.add_node('__end__', pos=np.mean([pos(e) for e in ends], axis=0), tip=True, r=0.0)
                ends_at[fork] = G.nodes['__end__']['pos'].copy()
                G.remove_node('__end__')
                path = path[:-1]
        js = [n for n in path if joint(n)]
        if not js:
            continue
        end = pos(path[-1]) if not joint(path[-1]) else ends_at.get(path[-1])
        last_j = js[-1]
        if end is None:
            kk = kids.get(last_j, [])
            end = np.mean([pos(c) for c in kk], axis=0) if kk else pos(last_j)
        if kind in PLANS:
            top, names, fr = PLANS[kind]
            use_top = top is not None and len(js) >= len(names) + 1
            first = 1 if use_top else 0
            chain = js[first:]
            pts = [pos(n) for n in chain] + [end]
            s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            f = s[1:-1] / max(s[-1], 1e-9)                       # the joints after the root joint
            # as many bones as the plan has names: one each, in order (a bird's leg: thigh, shin, the long foot, toes);
            # otherwise the joints nearest to where the plan puts them
            pick = {k: k for k in range(len(fr))} if len(chain) == len(names) else choose(list(f), list(fr))
            starts = {0: names[0]}
            for k, m in pick.items():
                starts[m + 1] = names[k + 1]
            if use_top:
                name[js[0]], role[js[0]] = top + sfx, kind
            seg = 0
            for m, n in enumerate(chain):
                if m in starts:
                    name[n], seg = starts[m] + sfx, 0
                else:
                    seg += 1
                    name[n] = f'{kind.capitalize()}Seg{m}{sfx}'
                role[n] = kind
        else:
            base = limb.get('base', 'Limb')
            for m, n in enumerate(js):
                name[n], role[n] = f'{base}_{m + 1}{sfx}', 'chain'
        for k, n in enumerate(js):
            tail[n] = pos(js[k + 1]) if k + 1 < len(js) else end
    # everything else hangs from a named bone: its name, a running number; the parent's role
    order = list(nx.bfs_tree(G, hips))
    count = {}
    order = [n for n in order if n not in gone]
    for n in order:
        if not joint(n) or n in name:
            continue
        p = par[n]
        while p not in name:
            p = par[p]
        pname = name[p]
        sfx = pname[-2:] if pname[-2:] in ('.L', '.R') else ''
        stem = pname[:-2] if sfx else pname
        count[stem, sfx] = count.get((stem, sfx), 0) + 1
        name[n], role[n] = f'{stem}X{count[stem, sfx]}{sfx}', role[p]
    for n in order:
        if not joint(n) or tail.get(n) is not None:
            continue
        kk = kids.get(n, [])
        if kk:
            tail[n] = np.mean([pos(c) for c in kk], axis=0)
        else:
            a = pos(par[n]) if n in par else pos(n) - np.array([0, 0, .05 * h])
            tail[n] = pos(n) + (pos(n) - a) * .3
    root_head = np.array([pos(hips)[0], pos(hips)[1], mesh.floor])
    bones = [dict(name='Root', head=root_head, tail=root_head + [0, 0, .1 * h], parent=None, role='root')]
    for n in order:
        if not joint(n):
            continue
        parent = 'Root' if n == hips else name[par[n]]
        bones.append(dict(name=name[n], head=pos(n), tail=np.asarray(tail[n], float), parent=parent, role=role[n]))
    names_seen = {}
    for b in bones:
        if b['name'] in names_seen:
            names_seen[b['name']] += 1
            b['name'] = f"{b['name']}_{names_seen[b['name']]}"
        else:
            names_seen[b['name']] = 0
    return bones


def kinds_for(A, body, taken=()):
    legs, ups = A['leg_pairs'], A['up_pairs']
    kinds = []
    if body in ('quadruped', 'centaur'):
        free = [k for k in ('hind', 'front') if k not in taken]
        kinds += list(zip(free, legs))
        rest_legs = legs[len(free):]
        kinds += [('arm', ups[0])] if ups else []
        rest_ups = ups[1:]
    elif body == 'winged':
        kinds += [('biped', legs[0])] if legs else []
        kinds += [('wing', ups[0])] if ups else []
        rest_legs, rest_ups = legs[1:], ups[1:]
    else:
        kinds += [('biped', legs[0])] if legs and 'biped' not in taken else []
        kinds += [('arm', ups[0])] if ups else []
        rest_legs = legs[1:] if legs and 'biped' not in taken else legs
        rest_ups = ups[1:]
    for k, p in enumerate(rest_legs + rest_ups):
        for l in p:
            l['base'] = f'Pair{k + 1}'
        kinds.append(('pair', p))
    for k, l in enumerate(A['singles']):
        l['base'] = f'Limb{k + 1}'
        kinds.append(('single', [l]))
    return kinds


# ----- the missing legs and the mirror ------------------------------------------------------------------------
def centre_line_bones(mesh, log):
    T = S.centre_tree(mesh.solid, mesh.lo, mesh.size)
    T = S.prune(T, .05 * mesh.height)
    T = S.collapse_tips(T, .15 * mesh.height)
    T = S.prune(T, .05 * mesh.height)
    A = S.analyse(T, mesh.height, mesh.facing, mid=mesh.mid, log=lambda m: None)
    return S.build(T, A, mesh.height, .01, log=lambda m: None)


def transplant(bones, mesh, missing, log):
    """For floor patches no bone reaches: the legs of the centre-line skeleton whose ends stand there, hung from the
    nearest trunk bone. Returns (new bones, the kinds they bring)."""
    ours = centre_line_bones(mesh, log)
    by = {b['name']: b for b in ours}
    kids = {}
    for b in ours:
        kids.setdefault(b['parent'], []).append(b['name'])
    limb_roles = ('hind', 'front', 'biped', 'chain')
    new, kinds, n_legs = [], set(), 0
    for b in ours:
        if b['role'] not in limb_roles or kids.get(b['name']):
            continue
        t = np.asarray(b['tail'])
        if t[2] > mesh.floor + .15 * mesh.height:
            continue
        if not any(np.linalg.norm(mesh.patches[k] - t[:2], axis=1).min() < .08 * mesh.height for k in missing):
            continue
        chain = [b]
        while chain[-1]['parent'] in by and by[chain[-1]['parent']]['role'] in limb_roles:
            chain.append(by[chain[-1]['parent']])
        chain.reverse()
        trunk = [x for x in bones if x['role'] in ('trunk', 'neck')]
        top = np.asarray(chain[0]['head'])
        near = min(trunk, key=lambda x: np.linalg.norm((np.asarray(x['head']) + np.asarray(x['tail'])) / 2 - top))
        taken = {x['name'] for x in bones} | {x['name'] for x in new}
        for k, x in enumerate(chain):
            y = dict(x, head=np.asarray(x['head'], float), tail=np.asarray(x['tail'], float))
            if y['name'] in taken:
                y['name'] = f"Leg{len(new) + 1}_{k + 1}{x['name'][-2:] if x['name'][-2:] in ('.L', '.R') else ''}"
                y['role'] = 'chain'
            y['parent'] = near['name'] if k == 0 else new[-1]['name']
            new.append(y)
        kinds.add(chain[0]['role'])
        n_legs += 1
        log(f"autorig: floor contact without a leg: the centre-line leg {chain[0]['name']}..{chain[-1]['name']} added")
    return new, kinds, n_legs


def symmetrize(bones, mesh, log):
    """Twin limbs, whole chains at a time: where the mesh around both is mirror-symmetric, each is made the exact
    mirror of the other (both moved to their mean); where it is not (a leg stepped forward, a hand round a sword) but
    the chains match bone for bone, only the bone lengths are made equal and each keeps its pose; a chain unlike its
    twin (a giant claw) stays as drawn. Middle bones go onto the middle plane where the mesh around them is
    symmetric."""
    by = {b['name']: b for b in bones}
    h = mesh.height
    pts = lambda b: np.array([b['head'] + (b['tail'] - b['head']) * s for s in np.linspace(0, 1, 5)])
    kids = {}
    for b in bones:
        kids.setdefault(b['parent'], []).append(b['name'])

    def chain(top):
        out, stack = [], [top]
        while stack:
            n = stack.pop(0)
            out.append(n)
            stack += kids.get(n, [])
        return out
    tops = [b['name'] for b in bones if b['name'].endswith('.L')
            and not str(by.get(b['parent'], {}).get('name', '')).endswith('.L')]
    mirrored, equal, kept = 0, 0, []
    for top in tops:
        left = chain(top)
        right = [n[:-2] + '.R' for n in left]
        if top[:-2] + '.R' not in by:
            continue
        if any(n not in by for n in right):
            kept.append(f'{top[:-2]} (other bones)')
            continue
        P_l = np.concatenate([pts(by[n]) for n in left])
        P_r = np.concatenate([pts(by[n]) for n in right])
        sym = min(mesh.symmetric_near(P_l), mesh.symmetric_near(P_r))
        if sym >= .8:
            for a, b in zip(left, right):
                for key in ('head', 'tail'):
                    m = (by[a][key] + mesh.mirror(by[b][key])) / 2
                    by[a][key], by[b][key] = m, mesh.mirror(m)
            mirrored += 1
            continue
        L = lambda n: float(np.linalg.norm(by[n]['tail'] - by[n]['head']))
        # a twin unlike it overall (one giant claw) stays; bone by bone they may differ more (where UniRig put a
        # collarbone's end)
        total = sum(L(a) for a in left), sum(L(b) for b in right)
        ratio = min(total) / max(max(total), 1e-9)
        if ratio < .75 or min(min(L(a), L(b)) / max(L(a), L(b), 1e-9) for a, b in zip(left, right)) < .4:
            kept.append(f'{top[:-2]} (mesh {sym:.2f}, lengths {ratio:.2f})')
            continue
        lengths = {}
        for a, b in zip(left, right):
            lengths[a] = lengths[b] = (L(a) + L(b)) / 2
        for side in (left, right):
            # each bone its twins' mean length along its own direction; a child that started at its parent's end
            # still does, one hung elsewhere on it moves with the parent's start
            moved = {}
            for n in side:
                bn = by[n]
                oh, ot = bn['head'].copy(), bn['tail'].copy()
                nh = oh
                if bn['parent'] in moved:
                    poh, pot, pnh, pnt = moved[bn['parent']]
                    nh = pnt.copy() if np.linalg.norm(oh - pot) < 1e-6 else oh + (pnh - poh)
                u = (ot - oh) / max(np.linalg.norm(ot - oh), 1e-9)
                bn['head'], bn['tail'] = nh, nh + u * lengths[n]
                moved[n] = (oh, ot, bn['head'], bn['tail'])
        equal += 1
    mids = 0
    for b in bones:
        if b['name'][-2:] in ('.L', '.R') or b['role'] not in ('root', 'trunk', 'neck', 'head', 'tail'):
            continue
        if max(abs(b['head'] @ mesh.left - mesh.mid), abs(b['tail'] @ mesh.left - mesh.mid)) > .08 * h:
            continue
        if mesh.symmetric_near(pts(b)) < .8:
            continue
        for key in ('head', 'tail'):
            b[key] = b[key] - ((b[key] @ mesh.left) - mesh.mid) * mesh.left
        mids += 1
    log(f'autorig: twin limbs: {mirrored} mirrored, {equal} made equal in length (posed differently)'
        f', {mids} middle bones on the middle plane' + (f'; as drawn: {", ".join(kept)}' if kept else ''))
    return mirrored + equal


def soles(bones, mesh, log):
    """A standing leg's last bone runs along its sole to the front of it: from its joint to where the toes end on
    the floor. UniRig drew a wolf's toes as a 2-4 cm stub straight down at the back of the paw; the reader's toe tip
    then sat under the wrist, nothing said where the paw pointed, and the paws landed and lifted tilted all through
    a walk. Each standing leg's end takes the floor-near skin closer to it than to any other leg's end."""
    h = mesh.height
    has_kids = {b['parent'] for b in bones}
    ends = [b for b in bones if b['name'] not in has_kids and b['role'] in ('hind', 'front', 'biped', 'pair', 'chain')
            and b['tail'][2] < mesh.floor + .05 * h]
    if not ends:
        return
    low = mesh.verts[mesh.verts[:, 2] < mesh.floor + .03 * h]
    seg_d = []
    for b in ends:
        a, c = b['head'][:2], b['tail'][:2]
        ab = c - a
        t = np.clip(((low[:, :2] - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        seg_d.append(np.linalg.norm(low[:, :2] - (a + t[:, None] * ab), axis=1))
    seg_d = np.array(seg_d)
    owner = np.argmin(seg_d, 0)
    moved = []
    for k, b in enumerate(ends):
        mine = low[(owner == k) & (seg_d[k] < .12 * h)]
        if len(mine) < 8:
            continue
        along = mine @ mesh.fwd
        tip = mine[along >= np.percentile(along, 97)].mean(0)
        if (tip - b['head']) @ mesh.fwd < .02 * h:            # the sole does not reach in front of the joint
            continue
        old = np.linalg.norm(b['tail'] - b['head'])
        b['tail'] = tip
        moved.append(f"{b['name']} {old / h * 100:.0f}->{np.linalg.norm(tip - b['head']) / h * 100:.0f}%")
    if moved:
        log(f'autorig: feet along their soles to the toe tips (length, % of height): {", ".join(moved)}')


def head_to_face(bones, mesh, log):
    """The head from the skull to the front of the face, on the middle plane, when UniRig ended the spine in an ear:
    a wolf's 'Head' was its left ear (3 cm, the neck bent toward it), the right ear hung beside it and nothing ran
    along the muzzle, so a howl lifted the muzzle by turning the ear and the face went sideways. Its siblings (ears,
    a jaw) hang from the new head."""
    h = mesh.height
    head = next((b for b in bones if b['role'] == 'head'), None)
    if head is None or abs(head['tail'] @ mesh.left - mesh.mid) < .04 * h:
        return
    by = {b['name']: b for b in bones}
    neck = by.get(head['parent'])
    sibs = [b for b in bones if b['parent'] == head['parent'] and b is not head
            and b['role'] in ('neck', 'head', 'chain') and np.linalg.norm(b['head'] - head['head']) < .25 * h]
    base = np.mean([head['head']] + [b['head'] for b in sibs], axis=0)
    base = base - ((base @ mesh.left) - mesh.mid) * mesh.left
    V = mesh.verts
    near = (np.linalg.norm(V - base, axis=1) < .35 * h) & (np.abs(V @ mesh.left - mesh.mid) < .04 * h)
    if near.sum() < 8:
        return
    along = V[near] @ mesh.fwd
    tip = V[near][along >= np.percentile(along, 98)].mean(0)
    if (tip - base) @ mesh.fwd < .04 * h:
        return
    if neck is not None and np.linalg.norm(neck['tail'] - head['head']) < 1e-6:
        neck['tail'] = base.copy()
    head['head'], head['tail'] = base, tip
    for k, b in enumerate(sibs):
        b['parent'] = head['name']
        old = b['name']
        sfx = old[-2:] if old[-2:] in ('.L', '.R') else ''
        b['name'], b['role'] = f'HeadX{k + 1}{sfx}', 'head'
        for c in bones:
            if c['parent'] == old:
                c['parent'] = b['name']
    # parents before children (Blender builds them in this order): the new children after the head
    bones[:] = [b for b in bones if not any(b is s for s in sibs)]
    at = next(k for k, b in enumerate(bones) if b is head) + 1
    bones[at:at] = sibs
    log(f'autorig: the head was off the middle (an ear?): now from the skull to the face, '
        f'{np.linalg.norm(tip - base) / h * 100:.0f}% of the height; {len(sibs)} bone(s) beside it hang from it')


def jaw_at_mouth(bones, mesh, log):
    """A mouth modelled open gets a jaw. On the middle plane, around the head bone: a gap with skin above and below
    it that is open to the front is the mouth; the Jaw runs from the gap's back end (the mouth's corner) to the
    front end of the skin under it (the chin) and hangs from the head. The reader reads the jaw's tip and the fit
    opens it (video2rig.reader: a bone named jaw), so a bite or a roar in the video opens the model's mouth.
    UniRig drew a lizard's jaw as a stub from inside the skull straight down (4 cm, its end under the throat):
    a bone on the middle plane ending below the mouth's corner is that stub and goes. A mouth modelled shut has no
    gap: nothing changes."""
    h, size = mesh.height, mesh.size
    head = next((b for b in bones if b['role'] == 'head'), None)
    if head is None or any('jaw' in b['name'].lower() for b in bones):
        return
    L = float(np.linalg.norm(head['tail'] - head['head']))
    fa, sa = int(np.argmax(np.abs(mesh.fwd))), int(np.argmax(np.abs(mesh.left)))
    if L < 4 * size or {fa, sa} != {0, 1}:
        return
    sf = 1.0 if mesh.fwd[fa] > 0 else -1.0
    k = int(np.clip(np.floor((mesh.mid - mesh.lo[sa]) / size), 1, mesh.solid.shape[sa] - 2))
    A = np.take(mesh.solid, [k - 1, k, k + 1], axis=sa).any(axis=sa)       # (forward axis, up): solid near the middle
    if sf < 0:
        A = A[::-1]
    nu, nz = A.shape
    f_of = lambda u: (mesh.lo[fa] + (u + .5) * size) if sf > 0 else (mesh.lo[fa] + (nu - u - .5) * size)
    # the head's own neighbourhood: nearer the head bone than 0.8 of its length, not far behind its joint
    U, Z = np.meshgrid(np.arange(nu), np.arange(nz), indexing='ij')
    P2 = np.stack([sf * np.array([f_of(u) for u in range(nu)])[U], mesh.lo[2] + (Z + .5) * size], -1)
    a2 = np.array([sf * head['head'][fa], head['head'][2]])
    b2 = np.array([sf * head['tail'][fa], head['tail'][2]])
    ab = b2 - a2
    t = np.clip(((P2 - a2) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    near = (np.linalg.norm(P2 - (a2 + t[..., None] * ab), axis=-1) < .8 * L) & (P2[..., 0] > a2[0] - .25 * L)
    gap = np.zeros_like(A)
    for u in range(nu):
        zs = np.nonzero(A[u] & near[u])[0]
        if len(zs) >= 2:
            gap[u, zs.min():zs.max()] = ~A[u, zs.min():zs.max()]
    lab, n = ndimage.label(gap)
    mouth = np.zeros_like(A)
    for c in range(1, n + 1):
        us, zs = np.nonzero(lab == c)
        front = us.max()
        open_front = front + 1 >= nu or not A[front + 1, zs[us == front]].all()
        if len(us) >= 6 and open_front and (front - us.min() + 1) * size >= .25 * L:
            mouth |= lab == c
    if not mouth.any():
        return
    us, zs = np.nonzero(mouth)
    ub, uf = us.min(), us.max()
    hinge_z = mesh.lo[2] + (zs[us == ub].mean() + .5) * size

    def under(u, top):                                     # the run of skin just under row `top` in column u
        z1 = top
        while z1 >= 0 and not A[u, z1]:
            z1 -= 1
        z0 = z1
        while z0 > 0 and A[u, z0 - 1]:
            z0 -= 1
        return (z0, z1) if z1 >= 0 else None

    run = under(uf, zs[us == uf].min())
    if run is None or (run[1] - run[0] + 1) * size > .7 * L:
        return                                              # the skin under the gap is no jaw: a chest, a shoulder
    u = uf
    while u + 1 < nu:                                       # on to the chin's front end
        nxt = np.nonzero(A[u + 1, max(run[0] - 1, 0):run[1] + 2])[0]
        if not len(nxt):
            break
        z = max(run[0] - 1, 0) + int(nxt[len(nxt) // 2])
        z0 = z1 = z
        while z0 > 0 and A[u + 1, z0 - 1]:
            z0 -= 1
        while z1 + 1 < nz and A[u + 1, z1 + 1]:
            z1 += 1
        if (z1 - z0 + 1) * size > .7 * L:
            break
        run, u = (z0, z1), u + 1
    hinge, tip = np.zeros(3), np.zeros(3)
    hinge[fa], hinge[2] = f_of(ub), hinge_z
    tip[fa], tip[2] = f_of(u) + sf * .5 * size, mesh.lo[2] + (.5 * (run[0] + run[1]) + .5) * size
    for p in (hinge, tip):
        p[sa] = mesh.mid
    if np.linalg.norm(tip - hinge) < .3 * L or np.linalg.norm(tip - head['tail']) > .7 * L:
        return
    has_kids = {b['parent'] for b in bones}
    stubs = [b for b in bones if b is not head and b['parent'] == head['name'] and b['role'] == 'head'
             and b['name'] not in has_kids and abs(b['tail'] @ mesh.left - mesh.mid) < .04 * h
             and abs(b['head'] @ mesh.left - mesh.mid) < .04 * h and b['tail'][2] < hinge_z]
    bones[:] = [b for b in bones if not any(b is s for s in stubs)]
    at = next(i for i, b in enumerate(bones) if b is head) + 1
    bones.insert(at, dict(name='Jaw', parent=head['name'], role='head', head=hinge, tail=tip))
    log(f'autorig: an open mouth: a jaw from its corner to the chin, {np.linalg.norm(tip - hinge) / L * 100:.0f}% '
        f'of the head\'s length' + (f'; {len(stubs)} stub(s) UniRig drew into it dropped' if stubs else ''))


def lone_head_ends(bones, mesh, log):
    """A short bone beside the head with no twin on the other side goes: UniRig drew a lizard one brow ridge
    (2 cm, on the left only). One-sided, it would take the left half of the face from the head and give the fit
    three more turns to guess with nothing in the video to read them from. Ears and horns come in pairs and stay."""
    h = mesh.height
    head = next((b for b in bones if b['role'] == 'head'), None)
    if head is None:
        return
    has_kids = {b['parent'] for b in bones}
    sibs = [b for b in bones if b is not head and b['parent'] == head['name'] and b['role'] == 'head'
            and b['name'] not in has_kids]
    lone = []
    for b in sibs:
        off = b['tail'] @ mesh.left - mesh.mid
        if abs(off) < .04 * h or np.linalg.norm(b['tail'] - b['head']) > DIGIT * h:
            continue
        mirrored = b['tail'] - 2 * off * mesh.left
        if not any(o is not b and np.linalg.norm(o['tail'] - mirrored) < .06 * h for o in sibs):
            lone.append(b)
    if lone:
        bones[:] = [b for b in bones if not any(b is s for s in lone)]
        log(f'autorig: {len(lone)} short bone(s) on one side of the head only, no twin: dropped')


# ----- one work folder ------------------------------------------------------------------------------------------
def score_draw(draw, mesh, log):
    bones = to_model(draw, mesh, log)
    G, root = joint_tree(bones)
    G, root, folded, dropped = clean(G, root, mesh)
    miss = mesh.uncovered(segments(G))
    gap = mirror_gap(G, mesh)
    # the share of the body farther than 15% of its size (its longest side: a lizard is three times as long as
    # high) from every bone: a draw that leaves a third of the body without a bone is no skeleton of it and counts
    # like a floor contact missed (of a robed figure's four draws, one had nothing below the waist: 45% of the skin
    # far from a bone, the others 8-10%; it won on the mirror gap)
    pts = np.array([a + (b - a) * t for a, b in segments(G) for t in np.linspace(0, 1, 11)])
    far = float((cKDTree(pts).query(mesh.vox[::4])[0] > .15 * float(np.max(mesh.hi_m - mesh.lo_m))).mean())
    return dict(G=G, root=root, missing=miss, gap=gap, folded=folded, dropped=dropped, far=far,
                score=len(miss) * 100 + gap * 100 + (100 if far > .33 else 0))


def finish(work, draws, log, facing='auto'):
    work = Path(work)
    mesh = Mesh(work, facing)
    h = mesh.height
    cands = []
    for k, p in enumerate(draws):
        out = work / f'unirig-draw{k}.json'
        cands.append(json.loads(out.read_text()) if out.exists() else None)
    scored = []
    for k, d in enumerate(cands):
        if d is None:
            continue
        try:
            s = score_draw(d, mesh, log)
        except Exception as e:                                  # noqa: BLE001 - a broken draw is skipped
            log(f'autorig: draw {k} unusable ({e})')
            continue
        log(f"autorig: draw {k}: {G_size(s['G'])} joints, floor contacts unreached {len(s['missing'])}/"
            f"{len(mesh.patches)}, mirror gap p90 {s['gap'] * 100:.1f}% of height"
            + (f", {100 * s['far']:.0f}% of the body far from every bone" if s['far'] > .33 else ''))
        scored.append((s['score'], k, s))
    if not scored:
        raise RuntimeError('no usable UniRig draw')
    _, k, s = min(scored, key=lambda x: x[0])
    G = s['G']
    A = S.analyse(G, h, mesh.facing, mid=mesh.mid, log=log, mid_tol=.12)
    hips = A['spine'][A['leg_pairs'][0][0]['at']] if A['leg_pairs'] else A['spine'][0]
    if G.nodes[hips]['tip']:
        hips = next(iter(G.neighbors(hips)))
    # missing legs first (they decide the body plan: two leg pairs and arms = a centaur)
    probe = label(G, hips, A, mesh, kinds_for(A, A['body']), log)
    new, taken, n_legs = [], set(), 0
    if s['missing']:
        new, taken, n_legs = transplant(probe, mesh, s['missing'], log)
    body = A['body']
    if body in ('biped', 'winged') and A['up_pairs']:
        # wings or arms: a flat membrane across the limb or a round arm (the shape, not the pose: a T-posed man spreads
        # his arms as wide as a bat its wings, a giant's arms reach lower than a wyvern's wings are long)
        dense = lambda path: np.concatenate([np.linspace(G.nodes[a]['pos'], G.nodes[b]['pos'], 20)
                                             for a, b in zip(path, path[1:])])
        flat = mesh.flatness(dense(A['up_pairs'][0][0]['path'][1:]), dense(A['spine']))
        body = 'winged' if flat > 5 else 'biped'
        log(f'autorig: upper limbs {flat:.1f}x as wide as thick across: {body}')
    if new:
        pairs = len(A['leg_pairs']) + (n_legs + 1) // 2
        body = 'centaur' if pairs >= 2 and A['up_pairs'] else 'quadruped' if pairs >= 2 else body
    bones = label(G, hips, A, mesh, kinds_for(A, body, taken), log) + new
    head_to_face(bones, mesh, log)
    jaw_at_mouth(bones, mesh, log)
    lone_head_ends(bones, mesh, log)
    soles(bones, mesh, log)
    mirrored = symmetrize(bones, mesh, log)
    miss = mesh.uncovered([(b['head'], b['tail']) for b in bones])
    spec = {'body': body, 'facing': mesh.facing, 'height': h, 'left': [float(v) for v in mesh.left],
            'source': 'unirig', 'unirig_draw': k,
            'bones': [dict(b, head=[float(v) for v in b['head']], tail=[float(v) for v in b['tail']])
                      for b in bones]}
    (work / 'joints.json').write_text(json.dumps(spec, indent=1) + '\n', encoding='utf-8')
    pts = nx.Graph()
    for i, v in enumerate(mesh.verts[::max(1, len(mesh.verts) // 4000)]):
        pts.add_node(i, pos=v)
    S.sheet(pts, spec['bones'], work / 'skeleton-sheet.png', h)
    print(f'AUTORIG_SKELETON {body} (UniRig draw {k}): {len(bones)} bones, floor contacts reached '
          f'{len(mesh.patches) - len(miss)}/{len(mesh.patches)}, {mirrored} twin pairs mirrored, '
          f'{len(new)} centre-line leg bones added -> {work / "joints.json"}', flush=True)


def sheet_only(work):
    work = Path(work)
    spec = json.loads((work / 'joints.json').read_text())
    V = np.load(work / 'shell.npz')['verts']
    pts = nx.Graph()
    for i, v in enumerate(V[::max(1, len(V) // 4000)]):
        pts.add_node(i, pos=v)
    S.sheet(pts, spec['bones'], work / 'skeleton-sheet.png', spec['height'])
    print(f"AUTORIG_SKELETON {spec['body']}: {len(spec['bones'])} bones -> {work / 'joints.json'}", flush=True)


def G_size(G):
    return sum(1 for n in G.nodes if not G.nodes[n]['tip'])


def main():
    ap = argparse.ArgumentParser(prog='video2rig.autorig.unirig')
    ap.add_argument('works', nargs='+', help='auto-rig work folders (blender/autorig_prepare.py made them)')
    ap.add_argument('--draws', type=int, default=4)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--unirig', default='', help='the UniRig folder (or set V2R_UNIRIG)')
    ap.add_argument('--blender', default='')
    ap.add_argument('--reuse', action='store_true', help='score the draws already in the work folders again')
    ap.add_argument('--facing', default='auto', choices=['auto', *S.AXES],
                    help='the way the creature faces (default: from its shape)')
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    blender = find_blender(a.blender)
    works = [str(Path(w).resolve()) for w in a.works]
    # many legs (a spider, a crab): traced from the body by rules, UniRig drew a spider's legs only to the knees
    many = [w for w in works if manylegs.build(w, log)]
    for w in many:
        spec = json.loads((Path(w) / 'joints.json').read_text())
        bones = [dict(b, head=np.array(b['head']), tail=np.array(b['tail'])) for b in spec['bones']]
        symmetrize(bones, Mesh(w, a.facing), log)
        spec['bones'] = [dict(b, head=[float(v) for v in b['head']], tail=[float(v) for v in b['tail']])
                         for b in bones]
        (Path(w) / 'joints.json').write_text(json.dumps(spec, indent=1) + '\n', encoding='utf-8')
        sheet_only(w)
    works = [w for w in works if w not in many]
    if not works:
        return
    if not a.reuse:
        unirig = find_unirig(a.unirig)
        if unirig is None:
            log('autorig: UniRig not found (--unirig or V2R_UNIRIG): the centre-line skeleton instead')
            for w in works:
                subprocess.run([sys.executable, '-m', 'video2rig.autorig.skeleton', w], check=True, cwd=REPO)
            return
        fbx = draw_all(works, a.draws, a.seed, unirig, blender, log)
        for w in works:
            for old in Path(w).glob('unirig-draw*.json'):
                old.unlink()
            got = fbx[w]
            if got:
                blender_io(blender, ['read', Path(w) / 'unirig-draws.json'] + got, py=unirig_python(unirig))
                for k, d in enumerate(json.loads((Path(w) / 'unirig-draws.json').read_text())):
                    (Path(w) / f'unirig-draw{k}.json').write_text(json.dumps(d), encoding='utf-8')
    for w in works:
        draws = sorted(Path(w).glob('unirig-draw[0-9]*.json'))
        if not draws:
            log(f'autorig: no UniRig draw for {w}: the centre-line skeleton instead')
            subprocess.run([sys.executable, '-m', 'video2rig.autorig.skeleton', w], check=True, cwd=REPO)
            continue
        finish(w, draws, log, a.facing)


if __name__ == '__main__':
    main()
