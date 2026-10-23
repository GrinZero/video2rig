"""A Seedance clip: its frames, its panels (views) with their cameras, the content timing and the silhouettes.

Cameras come from start/cameras.json (blender/render_start.py or export_camera.py), in world space; the rig lives
in the armature's space, so the armature's world matrix is folded into every camera. Seedance re-frames the start
image a little (a few percent of scale and offset, sometimes a turn of 10+ degrees), so every view is aligned on
frame 0, where the pose is known: scale, offset and a small rotation about the subject (orthographic cameras).
"""
import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .common import write_json


# ----- frames and timing -------------------------------------------------------------------------------------
def read_frames(path):
    """Every coded frame, as stored (a 24 fps clip in a 60 fps container decodes to its own 97 frames)."""
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        out.append(img)
    cap.release()
    if not out:
        raise SystemExit(f'cannot read {path}')
    return out


def cadence(frames, masks):
    """Content time of every frame. Seedance clips hold 24 fps content, or 30 fps with every 5th frame dropped, or
    25 with every 25th dropped, whatever the container says: a dropped content frame shows as one step in 4 (or 24)
    carrying about twice the motion (optical flow on the subject, divided by its local trend)."""
    small = [cv2.cvtColor(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA),
                          cv2.COLOR_BGR2GRAY) for f in frames]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    step = []
    for i in range(len(small) - 1):
        m = cv2.resize((masks[i] | masks[i + 1]).astype(np.uint8), small[i].shape[::-1]) > 0
        f = dis.calc(small[i], small[i + 1], None)
        step.append(float(np.linalg.norm(f[m], axis=1).mean()) if m.any() else 0.0)
    step = np.array(step)
    trend = np.array([np.median(step[max(0, i - 3):i + 4]) for i in range(len(step))])
    rel = step / np.maximum(trend, 1e-3)
    moving = trend > 0.5
    k = np.arange(len(step))
    by4 = [float(rel[moving & (k % 4 == r)].mean()) if (moving & (k % 4 == r)).any() else 1.0 for r in range(4)]
    m24 = moving & (k % 24 == 23)
    rec = {'rel_by_mod4': [round(v, 3) for v in by4],
           'rel_mod24_eq23': round(float(rel[m24].mean()), 3) if m24.any() else None}
    i = np.arange(len(frames))
    if max(by4) > 1.45 and sorted(by4)[-2] < 1.2:
        rec['content'] = '30fps_drop_every_5th'
        t = (i + i // 4) / 30.0
    elif rec['rel_mod24_eq23'] and rec['rel_mod24_eq23'] > 1.5:
        rec['content'] = '25fps_drop_every_25th'
        t = (i + i // 24) / 25.0
    else:
        rec['content'] = '24fps'
        t = i / 24.0
    return np.round(t, 6), rec


# ----- silhouettes --------------------------------------------------------------------------------------------
def background(frames):
    """The empty studio: the 85th percentile of every pixel over the take (the subject is darker than the backdrop
    and uncovers most pixels at some point), with a smooth cubic fit of the backdrop where it never moved away."""
    st = np.stack(frames[::2]).astype(np.float32)
    p85 = np.percentile(st, 85, axis=0)
    h, w = p85.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    u, v = xs / w, ys / h
    basis = np.stack([u ** a * v ** b for a in range(4) for b in range(4 - a)], -1).reshape(-1, 10)
    chroma = p85.max(2) - p85.min(2)
    keep = (chroma < np.percentile(chroma, 60) + 4).ravel()
    for _ in range(2):
        coef = np.linalg.lstsq(basis[keep], p85.reshape(-1, 3)[keep], rcond=None)[0]
        smooth = (basis @ coef).reshape(h, w, 3)
        diff = np.abs(p85 - smooth).max(2)
        keep = (diff < 10).ravel()
    out = np.where((np.abs(p85 - smooth).max(2) > 10)[..., None], smooth, p85)
    return out.astype(np.float32)


SHADE_LUM = np.linspace(-3.0, 0.5, 36)        # bins of log(brightness / backdrop's)
SHADE_HUE = np.linspace(0.0, 12.0, 25)        # bins of the colour's distance from the backdrop's (chromaticity x 100)


def shade_features(img, bg):
    """Per pixel: log of its brightness against the backdrop's, and how far its colour (brightness taken out) is from
    the backdrop's. A shadow only darkens the floor: negative brightness, colour distance near 0."""
    im = img.astype(np.float32)
    lum = np.log((im.mean(2) + 1.0) / (bg.mean(2) + 1.0))
    hue = np.abs(im / (im.sum(2, keepdims=True) + 3.0) - bg / (bg.sum(2, keepdims=True) + 3.0)).max(2) * 100.0
    return lum, hue


def learn_shadow(img, bg, mask, model):
    """The clip's own shadows, learned on frame 0 where the pose is known: the foreground the start pose does not
    cover is shadow (a contact shadow can be as dark as the body and as sharp), the pose's own pixels are the subject.
    A table over (brightness, colour distance): a cell is shadow when the off-pose foreground is four times as dense
    there as the subject, and only cells darker than the backdrop with nearly its colour can be shadow, so a coloured
    or bright body part never is. None when there is too little off-pose foreground to learn from."""
    subj = cv2.erode(model.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    rest = mask & ~cv2.dilate(model.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if subj.sum() < 200 or rest.sum() < max(200, .03 * subj.sum()):
        return None
    lum, hue = shade_features(img, bg)
    hs = cv2.GaussianBlur(np.histogram2d(lum[subj], hue[subj], (SHADE_LUM, SHADE_HUE))[0], (0, 0), 1) / subj.sum()
    hr = cv2.GaussianBlur(np.histogram2d(lum[rest], hue[rest], (SHADE_LUM, SHADE_HUE))[0], (0, 0), 1) / rest.sum()
    lc = .5 * (SHADE_LUM[:-1] + SHADE_LUM[1:])
    hc = .5 * (SHADE_HUE[:-1] + SHADE_HUE[1:])
    plausible = (lc[:, None] < -.05) & (hc[None, :] < 4.0)
    table = (hr > 4 * hs) & (hr > 2e-3) & plausible
    return table if table.any() else None


def silhouette(img, bg, shade=None):
    """Subject (and props) against the grey studio: colour difference from the backdrop, minus soft shadows (grey,
    darker than the floor, no colour cast), minus the clip's own shadows when learned (learn_shadow), and detached
    shadow patches; small enclosed holes filled."""
    im = img.astype(np.float32)
    diff = np.abs(im - bg).max(2)
    chroma = im.max(2) - im.min(2)
    chroma_bg = bg.max(2) - bg.min(2)
    lum, lum_bg = im.mean(2), bg.mean(2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.GaussianBlur(np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)), (0, 0), 2)
    shadow = (lum < lum_bg) & (np.abs(chroma - chroma_bg) < 6) & (diff < 45) & (tex < 25)
    if shade is not None:
        sl, sh = shade_features(img, bg)
        shadow |= shade[np.clip(np.digitize(sl, SHADE_LUM) - 1, 0, len(SHADE_LUM) - 2),
                        np.clip(np.digitize(sh, SHADE_HUE) - 1, 0, len(SHADE_HUE) - 2)]
    raw = (((diff > 14) | (chroma - chroma_bg > 12)) & ~shadow).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if n <= 1:
        return raw.astype(bool)
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    # every piece's means at once (a learned shadow can leave thousands of specks: one pass over the image each was
    # minutes a clip)
    flat = lab.ravel()
    cnt = np.maximum(np.bincount(flat, minlength=n), 1)
    mean = lambda a: np.bincount(flat, weights=a.ravel(), minlength=n) / cnt
    shadowy = (mean(chroma - chroma_bg) < 6) & (mean((lum < lum_bg).astype(np.float64)) > .8) & (mean(tex) < 20)
    keep = ~shadowy & (stats[:, cv2.CC_STAT_AREA] >= max(40, .005 * stats[big, cv2.CC_STAT_AREA]))
    keep[0], keep[big] = False, True
    mask = keep[lab]
    inv = (~mask).astype(np.uint8)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv, 4)
    H, W = mask.shape
    x, y, w_, h_, area = stats2.T
    hole = (x > 0) & (y > 0) & (x + w_ < W) & (y + h_ < H) & (area < 0.002 * H * W)
    hole[0] = False
    return mask | hole[lab2]


# ----- panels -------------------------------------------------------------------------------------------------
def turn_points(p, k, w, h):
    """Points (..., 2) as (column, row) in an image w wide and h tall -> the same points in np.rot90(image, k)
    (continuous pixels: column j covers [j, j + 1))."""
    p = np.asarray(p, float)
    u, v = p[..., 0], p[..., 1]
    k %= 4
    if k == 0:
        return np.stack([u, v], -1)
    if k == 1:
        return np.stack([v, w - u], -1)
    if k == 2:
        return np.stack([w - u, h - v], -1)
    return np.stack([h - v, u], -1)


class Panel:
    """Where one camera's image lies in the clip's frame. cameras.json gives the panel's corner (x0_px, y0_px, start
    frame pixels) and the camera's own image (width_px x height_px); `turn` quarter turns (np.rot90's k) take the
    rectangle in the frame to the camera's image. Side-by-side panels are full-height columns (no y0_px, no turn).
    A stacked frame (blender/render_stack.py: the side view above, the view from above below, the forward axis
    running the same way in both) keeps its view from above lying in the frame and turns it upright here, as the
    reader learned it from the side|top frame."""

    def __init__(self, rec, video_size, frame_px):
        W, H = video_size
        sx, sy = W / frame_px[0], H / frame_px[1]
        self.turn = int(rec.get('turn', 0)) % 4
        cw, ch = rec['width_px'], rec['height_px']
        fw, fh = (ch, cw) if self.turn % 2 else (cw, ch)        # the rectangle in the frame, start frame pixels
        self.x0, self.y0 = rec['x0_px'] * sx, rec.get('y0_px', 0) * sy
        self.fw, self.fh = fw * sx, fh * sy                     # the rectangle, video pixels (exact)
        self.x0i, self.y0i = int(round(self.x0)), int(round(self.y0))
        self.w = int(round(self.x0 + self.fw)) - self.x0i
        self.h = min(H, int(round(self.y0 + self.fh))) - self.y0i
        # the camera's image, video pixels
        self.cw, self.ch = (self.fh, self.fw) if self.turn % 2 else (self.fw, self.fh)
        win = rec.get('read_window')
        self.window = None if win is None else (win[0] * (sy if self.turn % 2 else sx),
                                                win[1] * (sx if self.turn % 2 else sy))

    def cut(self, img):
        """The camera's image out of a frame (or a mask) of the clip."""
        return np.rot90(img[self.y0i:self.y0i + self.h, self.x0i:self.x0i + self.w], self.turn)

    def put(self, full, part):
        """A camera image (or a mask) back into its place in a frame-sized array."""
        full[self.y0i:self.y0i + self.h, self.x0i:self.x0i + self.w] = np.rot90(part, -self.turn)

    def to_camera(self, p):
        """Points in a cut() image -> the camera's pixels (the cut starts at whole pixels, the camera at x0, y0)."""
        cut_w, cut_h = (self.h, self.w) if self.turn % 2 else (self.w, self.h)
        q = turn_points(p, -self.turn, cut_w, cut_h)
        q = q + np.array([self.x0i - self.x0, self.y0i - self.y0])
        return turn_points(q, self.turn, self.fw, self.fh)

    def from_camera(self, p):
        """The camera's pixels -> points in a cut() image."""
        cam_w, cam_h = self.cw, self.ch
        q = turn_points(p, -self.turn, cam_w, cam_h)
        q = q - np.array([self.x0i - self.x0, self.y0i - self.y0])
        return turn_points(q, self.turn, self.w, self.h)


# ----- cameras ------------------------------------------------------------------------------------------------
class Camera:
    """One panel's camera, mapping ARMATURE-space points to the panel's pixels (continuous: pixel column j covers
    [j, j + 1)). Correction (alignment on frame 0): scale s, pixel offset, a rotation rv (rotation vector) of the
    camera about a pivot point, so the pivot stays where it was."""

    def __init__(self, rec, video_size, frame_px, arm_world, pivot):
        sx = video_size[0] / frame_px[0]
        sy = video_size[1] / frame_px[1]
        self.panel = Panel(rec, video_size, frame_px)
        self.x0 = rec['x0_px'] * sx
        self.w, self.h = self.panel.cw, self.panel.ch
        self.size = (int(round(self.w)), int(round(self.h)))
        self.type = rec['type']
        self.loc = np.array(rec['location'], float)
        self.right0, self.up0 = np.array(rec['right'], float), np.array(rec['up'], float)
        self.fwd0 = np.array(rec['forward'], float)
        if self.type == 'ortho':
            self.k0 = self.w / rec['ortho_width']
        else:
            self.f0 = rec['focal_px'] * sx
        self.shift = np.array(rec.get('shift_px', (0.0, 0.0))) * np.array([sx, sy])
        M = np.array(arm_world, float)
        self.Ma, self.ta = M[:3, :3], M[:3, 3]
        self.pivot = self.Ma @ pivot + self.ta
        self.set(1.0, (0.0, 0.0), (0.0, 0.0, 0.0))

    def set(self, s, offset, rv):
        self.s, self.offset, self.rv = float(s), np.array(offset, float), np.array(rv, float)
        R = Rotation.from_rotvec(self.rv).as_matrix()
        self.right, self.up, self.fwd = R @ self.right0, R @ self.up0, R @ self.fwd0
        if self.type == 'ortho':
            k = self.k0 * self.s
            A0 = np.stack([self.right0, -self.up0]) * self.k0
            c = np.array([self.w / 2, self.h / 2]) + self.shift
            pivot_px = A0 @ (self.pivot - self.loc) + c           # where the uncorrected camera puts the pivot
            self.A = np.stack([self.right, -self.up]) * k @ self.Ma
            self.b = pivot_px + self.offset - np.stack([self.right, -self.up]) * k @ self.pivot \
                + np.stack([self.right, -self.up]) * k @ self.ta
        return self

    def project(self, X, jac=False):
        X = np.atleast_2d(X)
        if self.type == 'ortho':
            pix = X @ self.A.T + self.b
            return (pix, np.broadcast_to(self.A, (len(X), 2, 3))) if jac else pix
        Xw = X @ self.Ma.T + self.ta
        d = Xw - self.loc
        x, y, z = d @ self.right, d @ self.up, np.maximum(d @ self.fwd, 1e-6)
        f = self.f0 * self.s
        c = np.array([self.w / 2, self.h / 2]) + self.shift + self.offset
        pix = np.stack([c[0] + f * x / z, c[1] - f * y / z], 1)
        if not jac:
            return pix
        Ju = f * (self.right[None] / z[:, None] - (x / z ** 2)[:, None] * self.fwd[None])
        Jv = -f * (self.up[None] / z[:, None] - (y / z ** 2)[:, None] * self.fwd[None])
        return pix, np.stack([Ju, Jv], 1) @ self.Ma

    def depth(self, X):
        return (np.atleast_2d(X) @ self.Ma.T + self.ta - self.loc) @ self.fwd

    @property
    def px_per_unit(self):
        """Pixels per armature unit at the subject (at the pivot's distance for a camera with perspective)."""
        if self.type == 'ortho':
            return float(np.linalg.norm(self.A[1]))
        z = max(float((self.pivot - self.loc) @ self.fwd), 1e-6)
        return float(self.f0 * self.s / z * np.linalg.norm(self.Ma[:, 1]))

    def lift(self, pix, depth):
        """Armature-space points that project to pix (n, 2) and lie at `depth` (n,) along the camera's axis, as
        depth() measures it (world units from the camera)."""
        pix = np.atleast_2d(np.asarray(pix, float))
        depth = np.broadcast_to(np.asarray(depth, float), (len(pix),))
        if self.type == 'ortho':
            X0 = (pix - self.b) @ np.linalg.pinv(self.A).T
            n = np.linalg.solve(self.Ma, self.fwd)                  # the line of sight in armature space
            n /= np.linalg.norm(n)
            gain = float((self.Ma @ n) @ self.fwd)
            t = (depth - self.depth(X0)) / gain
            return X0 + t[:, None] * n
        f = self.f0 * self.s
        c = np.array([self.w / 2, self.h / 2]) + self.shift + self.offset
        x = (pix[:, 0] - c[0]) / f
        y = -(pix[:, 1] - c[1]) / f
        Xw = self.loc + depth[:, None] * (self.fwd[None] + x[:, None] * self.right[None] + y[:, None] * self.up[None])
        return (Xw - self.ta) @ np.linalg.inv(self.Ma).T


def model_mask(cam, X, tris, rasterize):
    tri, _, _ = rasterize(cam.project(X), cam.depth(X), tris, cam.size)
    return tri >= 0


def align(cam, X, tris, video_mask, rasterize, turn=True, start=None):
    """Scale, offset (and a turn of an orthographic camera about the subject) on frame 0 so the start pose's
    silhouette lies on the video's. Seedance usually keeps the start image, so the camera as rendered comes first.
    When it does not match (the video re-framed the shot: closer, lower, from further round), a coarse search over
    turns, each framed by the silhouette's area and centre, gives the starts for a fit of the outlines (chamfer
    distances both ways); the best overlap wins, and never one worse than the camera as rendered. start: an earlier
    alignment of the same view (a cleaner outline of the same frame): fitted from there only."""
    dt_video = cv2.distanceTransform((~video_mask).astype(np.uint8), cv2.DIST_L2, 3)
    vb = video_mask & ~cv2.erode(video_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    vy, vx = np.nonzero(vb)
    sub = np.linspace(0, len(vy) - 1, min(len(vy), 1500)).astype(int)
    vy, vx = vy[sub], vx[sub]
    vys, vxs = np.nonzero(video_mask)
    can_turn = turn and cam.type == 'ortho'

    def framed(rv):
        """Scale from the two areas, then offset from the two centres, for this turn."""
        x = np.array([0.0, 0.0, 0.0, *rv])
        cam.set(1.0, (0.0, 0.0), rv if can_turn else (0, 0, 0))
        m = model_mask(cam, X, tris, rasterize)
        if not m.any():
            return None
        x[0] = 0.5 * np.log(len(vys) / m.sum())
        cam.set(np.exp(x[0]), (0.0, 0.0), rv if can_turn else (0, 0, 0))
        m = model_mask(cam, X, tris, rasterize)
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        x[1:3] = (vxs.mean() - xs.mean(), vys.mean() - ys.mean())
        return x

    varea = float(video_mask.sum())
    h, w = video_mask.shape
    border = np.zeros_like(video_mask)
    border[:3], border[-3:], border[:, :3], border[:, -3:] = True, True, True, True

    def resid(x):
        cam.set(np.exp(x[0]), x[1:3], x[3:6] if can_turn else (0, 0, 0))
        m = model_mask(cam, X, tris, rasterize)
        if not m.any():
            return np.full(2 * 400 + 2, 60.0)
        mb = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        my, mx = np.nonzero(mb)
        pick = np.linspace(0, len(my) - 1, 400).astype(int)
        a = dt_video[my[pick], mx[pick]]
        dt_model = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
        pick2 = np.linspace(0, len(vy) - 1, 400).astype(int)
        b = dt_model[vy[pick2], vx[pick2]]
        # the outlines alone can be cheated by a model that grows past the frame (its outline leaves the image):
        # the areas must agree and the model must not run into the frame's edge more than the video does
        area = (m.sum() / max(varea, 1.0) - 1.0) * 120.0
        edge = ((m & border).sum() - (video_mask & border).sum()) / 20.0
        return np.concatenate([np.minimum(np.concatenate([a, b]), 60.0), [area, max(edge, 0.0)]])

    def iou_of(x):
        cam.set(np.exp(x[0]), x[1:3], x[3:6] if can_turn else (0, 0, 0))
        m = model_mask(cam, X, tris, rasterize)
        return float((m & video_mask).sum() / max((m | video_mask).sum(), 1))
    raw = np.zeros(6)
    raw_iou = iou_of(raw)
    cands = [(raw_iou, raw)]
    if start is not None:
        x = np.array([np.log(start['scale']), *start['offset'], *np.radians(start['turn_deg'])])
        cands = [(iou_of(x), x)]
    x = framed((0.0, 0.0, 0.0)) if start is None else None
    if x is not None:
        cands.append((iou_of(x), x))
    if raw_iou < .9 and can_turn and start is None:
        # the video re-framed the shot: turns about the up axis (round the subject) and the camera's right axis
        # (higher, lower), every one framed by area and centre, scored by overlap
        up, right = np.array([0.0, 0.0, 1.0]), cam.right0
        for az in range(-30, 31, 6):
            for el in range(-12, 13, 6):
                if az == 0 and el == 0:
                    continue
                rv = (Rotation.from_rotvec(up * np.radians(az)) * Rotation.from_rotvec(right * np.radians(el))).as_rotvec()
                x = framed(rv)
                if x is not None:
                    cands.append((iou_of(x), x))
    cands.sort(key=lambda c: -c[0])
    best_iou, best_x = cands[0]
    for _, x0 in cands[:3]:
        lo = np.array([x0[0] - .25, x0[1] - 150, x0[2] - 150, *(x0[3:6] - .2)])
        hi = np.array([x0[0] + .25, x0[1] + 150, x0[2] + 150, *(x0[3:6] + .2)])
        sol = least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi), diff_step=[1e-3, .05, .05, 2e-3, 2e-3, 2e-3],
                            max_nfev=120)
        iou = iou_of(sol.x)
        if iou > best_iou:
            best_iou, best_x = iou, sol.x
    x, iou = best_x, best_iou
    # last, small steps on the overlap itself (the outline fit stops a little short of it)
    steps = np.array([.03, 3.0, 3.0, *[np.radians(2.0) * can_turn] * 3])
    for _ in range(3):
        for _ in range(12):
            better = False
            for k in np.nonzero(steps)[0]:
                for sign in (1.0, -1.0):
                    x2 = x.copy()
                    x2[k] += sign * steps[k]
                    i2 = iou_of(x2)
                    if i2 > iou + 1e-4:
                        x, iou, better = x2, i2, True
                        break
            if not better:
                break
        steps /= 2
    if iou < raw_iou + .005:                        # never worse than the camera as rendered
        x, iou = raw, raw_iou
    iou_of(x)
    return dict(scale=float(np.exp(x[0])), offset=[float(v) for v in x[1:3]],
                turn_deg=[float(np.degrees(v)) for v in x[3:6]], frame0_iou=round(iou, 4), raw_iou=round(raw_iou, 4))


def load_cameras(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    write_json(path, data)
