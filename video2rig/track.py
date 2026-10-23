"""Seed points on the clip's first frame (our own render) and track them through the whole clip.

Frame 0 of a Seedance clip is the render it started from, so every model pixel there is a known point of the mesh
(triangle + barycentric weights). Thin, fast parts (legs) get dense seeds, the rest sparse ones. BootsTAPIR
(Apache 2.0, vendored in third_party/) carries them through every frame, with a visibility estimate per frame.
"""
import json
import subprocess
import urllib.request

import cv2
import numpy as np

from .camera import depth as view_depth
from .camera import project
from .common import CACHE, CHECKPOINT_URL, TOOL, torch_python

CREATE_NO_WINDOW = 0x08000000


def checkpoint():
    path = CACHE / 'bootstapir_checkpoint_v2.pt'
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print('downloading the tracker checkpoint (219 MB) ->', path, flush=True)
        urllib.request.urlretrieve(CHECKPOINT_URL, str(path) + '.part')
        (CACHE / 'bootstapir_checkpoint_v2.pt.part').rename(path)
    return path


def dominant_bone(idx, wts):
    return idx[np.arange(len(idx)), np.argmax(wts, axis=1)]


def seed(creature, clip, view, cfg, pose=None, frame=0, dense_only=False):
    """Visible model pixels of one frame on a grid: dense on the parts listed in creature.parts().

    Frame 0 with the clip's first pose (the render Seedance started from) needs no fit; a later frame takes the
    pose of a first fit (re-seeding), and then only pixels inside the video's silhouette count."""
    rig, mesh = creature.rig, creature.mesh
    X = rig.skin(creature.p0(clip) if pose is None else pose, mesh['coords'], creature.idx, creature.wts)
    pix = project(view, X)
    w, h = view['size']
    tri, bary, _ = creature.rasterize(pix, view_depth(view, X), mesh['triangles'], view['size'])
    valid = tri >= 0
    inner = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    vmask = view['mask'](frame)
    inner &= cv2.erode(vmask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if hasattr(creature, 'render'):
        # Seedance's first frame is close to the start render, not always identical (a head already turned):
        # seed only where our render and the video look alike (patch correlation), or are both flat and similar.
        img, _ = creature.render(clip, X, view)
        r = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        v = cv2.cvtColor(view['frame'](frame), cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = (7, 7)
        mr, mv = cv2.blur(r, k), cv2.blur(v, k)
        sr = np.sqrt(np.maximum(cv2.blur(r * r, k) - mr * mr, 0))
        sv = np.sqrt(np.maximum(cv2.blur(v * v, k) - mv * mv, 0))
        ncc = (cv2.blur(r * v, k) - mr * mv) / np.maximum(sr * sv, 1e-3)
        flat = (sr < 4) & (sv < 4)
        alike = (ncc > float(cfg.get('seed_min_match', .25))) | flat
        inner &= alike
    bone_of_vertex = dominant_bone(creature.idx, creature.wts)
    tris = mesh['triangles']
    part_bones = {j: name for name, ids in creature.parts().items() for j in ids}
    vb = bone_of_vertex[tris]
    tri_bone = np.where(vb[:, 1] == vb[:, 2], vb[:, 1], vb[:, 0])      # majority of the three corners
    dense = np.isin(tri_bone, list(part_bones))
    ys, xs = np.mgrid[0:h, 0:w]
    # spacing in pixels of a view the tracker's size: a 1080p panel gets the same points as a 720p one (the tracker
    # sees 768 px either way; twice the points only doubled the time and the graphics memory)
    k = max(1.0, max(w, h) / float(cfg.get('track_side', 768)))
    s_dense = max(1, int(round(int(cfg.get('seed_dense_px', 2)) * k)))
    s_body = max(1, int(round(int(cfg.get('seed_body_px', 8)) * k)))
    t = np.where(tri >= 0, tri, 0)
    on_dense = dense[t] & (tri >= 0)
    grid = np.where(on_dense, (xs % s_dense == 0) & (ys % s_dense == 0), (xs % s_body == 0) & (ys % s_body == 0))
    sel = inner & grid
    if dense_only:
        sel &= on_dense
    yy, xx = np.nonzero(sel)
    out = dict(tri=tri[yy, xx], bary=bary[yy, xx], px=np.stack([xx + .5, yy + .5], 1).astype(np.float32),
               bone=tri_bone[tri[yy, xx]], dense=on_dense[yy, xx], t=np.full(len(yy), frame))
    print(f'seeds {clip}/{view["name"]} frame {frame}: {len(yy)} ({int(out["dense"].sum())} dense)', flush=True)
    return out


def square_video(view, n_frames, side):
    w, h = view['size']
    S = max(w, h)
    top = (S - h) // 2
    k = side / S
    frames = []
    for i in range(n_frames):
        f = cv2.cvtColor(view['frame'](i), cv2.COLOR_BGR2RGB)
        g = cv2.copyMakeBorder(f, top, S - h - top, 0, S - w, cv2.BORDER_REPLICATE)
        frames.append(cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC))
    return np.stack(frames), top, k


def other_trackers():
    """How many trackers are running on the graphics card (ours from another clip, or another session's): two at once
    overflow a 12 GB card, and then both crawl. Trackers on the processor do not count (they once held the card idle
    for an hour)."""
    import os
    try:
        if os.name == 'nt':
            cmd = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
                   "$_.CommandLine -like '*tracker_worker.py*gpu' } | Measure-Object | Select-Object -ExpandProperty Count")
            out = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, timeout=60,
                                 creationflags=CREATE_NO_WINDOW)
            return int(out.stdout.decode().strip() or 0)
        out = subprocess.run(['ps', '-eo', 'args'], capture_output=True, timeout=60)
        return sum('tracker_worker.py' in line and 'python' in line and line.rstrip().endswith('gpu')
                   for line in out.stdout.decode().splitlines())
    except Exception:
        return 0


class card_lock:
    """One tracker on the graphics card at a time, across terminals: an OS lock on a file in the cache (released by
    the OS when the process ends, so it never goes stale), then a wait for trackers that do not take it."""

    def __init__(self, device):
        self.device, self.f = device, None

    def __enter__(self):
        import os
        import time
        if self.device != 'gpu':
            return self
        CACHE.mkdir(parents=True, exist_ok=True)
        self.f = open(CACHE / 'gpu.lock', 'a+b')
        told = False
        while True:
            try:
                self.f.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if not told:
                    print('waiting for the graphics card: another tracker is running', flush=True)
                    told = True
                time.sleep(5)
        # trackers started without the lock: wait until none has run for 20 s (one that saw the card free at the
        # same moment shows up within that time)
        quiet = 0
        while quiet < 3:
            if other_trackers() > 0:
                if not told:
                    print('waiting for the graphics card: another tracker is running', flush=True)
                    told = True
                quiet = 0
            else:
                quiet += 1
            if quiet < 3:
                time.sleep(10)
        return self

    def __exit__(self, *exc):
        if self.f is None:
            return False
        import os
        try:
            self.f.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f, fcntl.LOCK_UN)
        finally:
            self.f.close()
        return False


def run_tracker(video, queries, work, device):
    job, out = work / 'tracker-job.npz', work / 'tracker-out.npz'
    np.savez(job, video=video, queries=queries)
    for attempt in (1, 2):
        with card_lock(device):
            r = subprocess.run([str(torch_python()), str(TOOL / 'video2rig/tracker_worker.py'), str(job), str(out),
                                str(checkpoint()), device], capture_output=True, creationflags=CREATE_NO_WINDOW)
        print(r.stdout.decode('utf-8', 'replace').strip(), flush=True)
        if r.returncode == 0:
            break
        err = r.stderr.decode('utf-8', 'replace')[-4000:]
        # a tracker that dies without a word (the graphics driver, not Python) usually runs the second time
        if attempt == 2 or err.strip():
            raise SystemExit(f'tracker failed (exit code {r.returncode}):\n' + err)
        print(f'the tracker stopped without a message (exit code {r.returncode}): once more', flush=True)
    job.unlink()
    return np.load(out)


def track(creature, clip, cfg, device='gpu', retrack=False):
    """Tracks for every view of the clip, cached in <project>/work/<clip>/tracks.npz."""
    work = creature.project.work(clip)
    path = work / 'tracks.npz'
    sig = json.dumps({**{k: cfg.get(k) for k in ('seed_dense_px', 'seed_body_px', 'track_side')},
                      'parts': creature.parts()}, sort_keys=True)
    if path.exists() and not retrack:
        d = np.load(path, allow_pickle=True)
        if str(d['signature']) == sig:
            return {k: d[k] for k in d.files}
    n_frames = len(creature.times(clip))
    side = int(cfg.get('track_side', 768))
    parts = {k: [] for k in ('view', 'tri', 'bary', 'px0', 'bone', 'dense', 'tracks', 'visible_p')}
    for vi, view in enumerate(creature.views(clip)):
        s = seed(creature, clip, view, cfg)
        video, top, k = square_video(view, n_frames, side)
        q = np.zeros((len(s['px']), 3), np.float32)
        q[:, 1] = (s['px'][:, 1] + top) * k - .5
        q[:, 2] = s['px'][:, 0] * k - .5
        res = run_tracker(video, q, work, device)
        tr = res['tracks'].astype(np.float64)
        tr[..., 0] = (tr[..., 0] + .5) / k
        tr[..., 1] = (tr[..., 1] + .5) / k - top
        sig_o = 1 / (1 + np.exp(-res['occlusion'].astype(np.float64)))
        sig_e = 1 / (1 + np.exp(-res['expected_dist'].astype(np.float64)))
        parts['view'].append(np.full(len(q), vi))
        parts['tri'].append(s['tri']); parts['bary'].append(s['bary']); parts['px0'].append(s['px'])
        parts['bone'].append(s['bone']); parts['dense'].append(s['dense'])
        parts['tracks'].append(tr.astype(np.float32)); parts['visible_p'].append(((1 - sig_o) * (1 - sig_e)).astype(np.float32))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out['seed_t'] = np.zeros(len(out['tri']), int)
    out['signature'] = np.array(sig)
    np.savez_compressed(path, **out)
    return out


def reseed(creature, clip, cfg, params, fit_path, device='gpu', retrack=False):
    """Second pass: dense seeds on later frames, placed by a first fit (its pose says which mesh point each pixel
    shows), tracked both ways through the clip. A part that hides for a while (the far leg crossing behind the
    near one) gets fresh points from just before and after, instead of only from frame 0."""
    work = creature.project.work(clip)
    path = work / 'tracks-reseed.npz'
    every = int(cfg.get('reseed_every', 12))
    T = len(creature.times(clip))
    frames = list(range(every, T, every))
    sig = json.dumps({'fit': str(fit_path), 'mtime': fit_path.stat().st_mtime, 'frames': frames,
                      'parts': creature.parts(), **{k: cfg.get(k) for k in ('seed_dense_px', 'track_side')}},
                     sort_keys=True)
    if path.exists() and not retrack:
        d = np.load(path, allow_pickle=True)
        if str(d['signature']) == sig:
            return {k: d[k] for k in d.files}
    side = int(cfg.get('track_side', 768))
    parts = {k: [] for k in ('view', 'tri', 'bary', 'px0', 'bone', 'dense', 'tracks', 'visible_p', 'seed_t')}
    for vi, view in enumerate(creature.views(clip)):
        seeds = [seed(creature, clip, view, cfg, pose=params[f], frame=f, dense_only=True) for f in frames]
        s = {k: np.concatenate([x[k] for x in seeds]) for k in seeds[0]}
        video, top, k = square_video(view, T, side)
        q = np.zeros((len(s['px']), 3), np.float32)
        q[:, 0] = s['t']
        q[:, 1] = (s['px'][:, 1] + top) * k - .5
        q[:, 2] = s['px'][:, 0] * k - .5
        res = run_tracker(video, q, work, device)
        tr = res['tracks'].astype(np.float64)
        tr[..., 0] = (tr[..., 0] + .5) / k
        tr[..., 1] = (tr[..., 1] + .5) / k - top
        sig_o = 1 / (1 + np.exp(-res['occlusion'].astype(np.float64)))
        sig_e = 1 / (1 + np.exp(-res['expected_dist'].astype(np.float64)))
        parts['view'].append(np.full(len(q), vi))
        parts['tri'].append(s['tri']); parts['bary'].append(s['bary']); parts['px0'].append(s['px'])
        parts['bone'].append(s['bone']); parts['dense'].append(s['dense']); parts['seed_t'].append(s['t'])
        parts['tracks'].append(tr.astype(np.float32)); parts['visible_p'].append(((1 - sig_o) * (1 - sig_e)).astype(np.float32))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out['signature'] = np.array(sig)
    np.savez_compressed(path, **out)
    return out


def merge(a, b):
    keys = ('view', 'tri', 'bary', 'px0', 'bone', 'dense', 'tracks', 'visible_p', 'seed_t')
    return {k: np.concatenate([a.get(k, np.zeros(len(a['tri']), int)) if k == 'seed_t' else a[k], b[k]]) for k in keys}
