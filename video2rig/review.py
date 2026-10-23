"""Comparison film of a clip: every view of the video, the fit rendered through that view's own camera and an
overlay of both outlines, plus a three-quarter view the video never had (depth, hidden limbs). Content timing on a
60 fps grid.

    python -m video2rig review <project> <clip> [--tag=-v2r]
"""
import subprocess

import cv2
import numpy as np

from .clip import Camera
from .common import ffmpeg

FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale):
    """Dark text on a light box."""
    (w, h), base = cv2.getTextSize(s, FONT, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 5), (x + w + 4, y + base + 3), (235, 235, 235), -1)
    cv2.putText(img, s, (x, y), FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)


def orbit(creature, clip, yaw_deg=50.0, pitch_deg=18.0):
    """An orthographic camera from three-quarter front, framing the whole clip's motion."""
    view = creature.views(clip)[0]
    cam0 = view['cam']
    rig, mesh = creature.rig, creature.mesh
    up = creature.up
    f0 = cam0.fwd @ cam0.Ma
    f0 = f0 - (f0 @ up) * up
    f0 /= np.linalg.norm(f0)
    side = np.cross(up, f0)
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    f = np.cos(yaw) * f0 + np.sin(yaw) * side
    f = np.cos(pitch) * f - np.sin(pitch) * up
    right = np.cross(f, up); right /= np.linalg.norm(right)
    upv = np.cross(right, f)
    return dict(f=f, right=right, up=upv)


def render_orbit(creature, X, o, size, centre, span):
    """Render posed vertices through an orthographic orbit view (armature space, no world matrix)."""
    w, h = size
    rec = {'x0_px': 0, 'width_px': w, 'height_px': h, 'type': 'ortho', 'location': list(centre - o['f'] * 10 * span),
           'right': list(o['right']), 'up': list(o['up']), 'forward': list(o['f']), 'ortho_width': span * 1.6 * w / h}
    cam = Camera(rec, (w, h), (w, h), np.eye(4), centre)
    return creature.render(None, X, cam)[0]


def review(creature, clip, fits, out_dir, mp4, scale=None):
    """fits: [(label, params [T, n])]; the first one is also shown from three-quarter view and as an overlay."""
    views = creature.views(clip)
    times = creature.times(clip)
    T = len(times)
    out_dir.mkdir(parents=True, exist_ok=True)
    rig, mesh = creature.rig, creature.mesh
    name0, P0 = fits[0]
    pts = np.concatenate([rig.skin(P0[i], mesh['coords'][::7], creature.idx[::7], creature.wts[::7]) for i in range(0, T, 6)])
    centre = (pts.min(0) + pts.max(0)) / 2
    span = float(np.ptp(pts @ creature.up)) * 1.15
    o = orbit(creature, clip)
    vh = views[0]['size'][1]
    scale = scale or min(1.0, 400 / vh)
    for i in range(T):
        rows = []
        Xs = [rig.skin(p[i], mesh['coords'], creature.idx, creature.wts) for _, p in fits]
        for v in views:
            w, h = int(v['size'][0] * scale), int(v['size'][1] * scale)
            vid = v['frame'](i)
            cells = [(vid, f'SEEDANCE {v["name"].upper()}' if len(views) > 1 else 'SEEDANCE')]
            for (label, _), X in zip(fits, Xs):
                cells.append((creature.render(clip, X, v)[0], label))
            img0, valid = creature.render(clip, Xs[0], v)
            over = (vid.astype(np.float32) * .5 + img0.astype(np.float32) * .5).astype(np.uint8)
            vm = v['mask'](i)
            cr, _ = cv2.findContours(valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv_, _ = cv2.findContours(vm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(over, cv_, -1, (255, 80, 0), 1)
            cv2.drawContours(over, cr, -1, (0, 0, 255), 1)
            iou = (valid & vm).sum() / max((valid | vm).sum(), 1)
            cells.append((over, f'OVERLAY  outline match {iou:.2f}'))
            row = []
            for img, label in cells:
                c = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                text(c, label, (8, 22), .5)
                row.append(c)
            text(row[0], f'frame {i:03d}  {times[i]:.2f} s', (8, h - 10), .45)
            rows.append(np.hstack(row))
        w3 = rows[0].shape[1] // (len(fits) + 2)
        h3 = rows[0].shape[0]
        three = render_orbit(creature, Xs[0], o, (w3 * 2, h3 * 2), centre, span)
        three = cv2.resize(three, (w3, h3), interpolation=cv2.INTER_AREA)
        text(three, f'{name0}: 3/4 VIEW', (8, 22), .5)
        rows.append(three)
        W = max(r.shape[1] for r in rows)               # panels of a split frame can differ in width
        rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), 245, np.uint8)]) for r in rows]
        cv2.imwrite(str(out_dir / f'{i:03d}.png'), np.vstack(rows))
    return film(out_dir, times, mp4)


def film(folder, times, mp4):
    T = len(times)
    lst = folder / 'frames.txt'
    with open(lst, 'w') as f:
        for i in range(T):
            dur = (times[i + 1] - times[i]) if i + 1 < T else 1 / 60
            f.write(f"file '{i:03d}.png'\nduration {dur:.6f}\n")
        f.write(f"file '{T - 1:03d}.png'\n")
    subprocess.run([ffmpeg(), '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(lst), '-vf',
                    'fps=60,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p', '-c:v', 'libx264', '-crf', '18', str(mp4)],
                   check=True, **({'creationflags': 0x08000000} if hasattr(subprocess, 'CREATE_NO_WINDOW') else {}))
    return mp4
