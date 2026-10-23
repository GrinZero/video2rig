"""A small z-buffer rasterizer and a textured render (no window): on the graphics card when PyTorch with CUDA is
installed (the tracker needs it anyway), else numpy on the processor; both give the same pixels.

Pixel coordinates are continuous: pixel column j covers [j, j + 1), its centre is j + 0.5.
"""
import os

import cv2
import numpy as np

LIGHTS = [((-3, -4, 5), .55), ((4, -2, 3), .35), ((0, 4, 4), .30)]
BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1 << 14]   # triangles grouped by their bounding box size, px
GPU_BUCKETS = [4, 16, 64, 256, 1 << 14]                # fewer, larger groups: the card's cost is the number of steps
_DEVICE = None


def device():
    """The CUDA device when PyTorch has one (unless V2R_CPU=1), else None (numpy)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = False
        if not os.environ.get('V2R_CPU'):
            try:
                import torch
                if torch.cuda.is_available():
                    _DEVICE = torch.device('cuda')
            except ImportError:
                pass
    return _DEVICE or None


def rasterize(pix, depth, tris, size):
    """Triangle id per pixel (-1 empty), barycentric weights and depth (smaller = nearer the camera)."""
    dev = device()
    if dev is not None:
        return rasterize_torch(pix, depth, tris, size, dev)
    return rasterize_numpy(pix, depth, tris, size)


def rasterize_torch(pix, depth, tris, size, dev):
    """rasterize on the graphics card, the same steps in float64: fragments of every triangle's bounding box grouped
    by size, then the nearest fragment per pixel (ties: the first, as numpy's stable sort keeps)."""
    import torch
    width, height = size
    P = torch.as_tensor(np.ascontiguousarray(pix), dtype=torch.float64, device=dev)
    D = torch.as_tensor(np.ascontiguousarray(depth), dtype=torch.float64, device=dev)
    F = torch.as_tensor(np.ascontiguousarray(tris), dtype=torch.int64, device=dev)
    p, z = P[F], D[F]
    x0 = torch.clamp(torch.ceil(p[:, :, 0].amin(1) - .5), min=0).long()
    x1 = torch.clamp(torch.floor(p[:, :, 0].amax(1) - .5), max=width - 1).long()
    y0 = torch.clamp(torch.ceil(p[:, :, 1].amin(1) - .5), min=0).long()
    y1 = torch.clamp(torch.floor(p[:, :, 1].amax(1) - .5), max=height - 1).long()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    a, bb, c = p[:, 0], p[:, 1], p[:, 2]
    area = (bb[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (bb[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    ok = (bw > 0) & (bh > 0) & (area.abs() > 1e-12)
    size_px = torch.maximum(bw, bh)
    frags = []

    def block(sel, S):
        oy = torch.arange(S * S, device=dev) // S
        ox = torch.arange(S * S, device=dev) % S
        px, py = x0[sel, None] + ox[None], y0[sel, None] + oy[None]
        inb = (ox[None] < bw[sel, None]) & (oy[None] < bh[sel, None])
        cx, cy = px + .5, py + .5
        A, B, C, ar = a[sel], bb[sel], c[sel], area[sel][:, None]
        w0 = ((B[:, 0, None] - cx) * (C[:, 1, None] - cy) - (B[:, 1, None] - cy) * (C[:, 0, None] - cx)) / ar
        w1 = ((C[:, 0, None] - cx) * (A[:, 1, None] - cy) - (C[:, 1, None] - cy) * (A[:, 0, None] - cx)) / ar
        w2 = 1 - w0 - w1
        inside = inb & (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        r, q = torch.nonzero(inside, as_tuple=True)
        if not len(r):
            return
        t = sel[r]
        ww = torch.stack([w0[r, q], w1[r, q], w2[r, q]], 1)
        frags.append((py[r, q] * width + px[r, q], (ww * z[t]).sum(1), t, ww))

    lo = 0
    for S in GPU_BUCKETS:
        sel = torch.nonzero(ok & (size_px > lo) & (size_px <= S)).squeeze(1)
        lo = S
        if not len(sel):
            continue
        if S > 256:                                        # huge triangles: one at a time, their own box
            for t in sel.tolist():
                block(sel.new_tensor([t]), int(size_px[t]))
            continue
        step = max(1, (1 << 22) // (S * S))                # about 4M fragments a block
        for k in range(0, len(sel), step):
            block(sel[k:k + step], S)
    WH = width * height
    tri = torch.full((WH,), -1, dtype=torch.int64, device=dev)
    bary = torch.zeros((WH, 3), dtype=torch.float64, device=dev)
    zbuf = torch.full((WH,), float('inf'), dtype=torch.float64, device=dev)
    if frags:
        Pp = torch.cat([f[0] for f in frags])
        Z = torch.cat([f[1] for f in frags])
        T = torch.cat([f[2] for f in frags])
        WW = torch.cat([f[3] for f in frags])
        zbuf.scatter_reduce_(0, Pp, Z, 'amin')
        win = Z <= zbuf[Pp]
        order = torch.arange(len(Pp), device=dev)
        first = torch.full((WH,), len(Pp), dtype=torch.int64, device=dev)
        first.scatter_reduce_(0, Pp[win], order[win], 'amin')
        has = first < len(Pp)
        tri[has] = T[first[has]]
        bary[has] = WW[first[has]]
    return (tri.reshape(height, width).cpu().numpy(), bary.reshape(height, width, 3).cpu().numpy(),
            zbuf.reshape(height, width).cpu().numpy())


def rasterize_numpy(pix, depth, tris, size):
    width, height = size
    p = pix[tris]
    z = depth[tris]
    x0 = np.maximum(np.ceil(p[:, :, 0].min(1) - .5).astype(np.int64), 0)
    x1 = np.minimum(np.floor(p[:, :, 0].max(1) - .5).astype(np.int64), width - 1)
    y0 = np.maximum(np.ceil(p[:, :, 1].min(1) - .5).astype(np.int64), 0)
    y1 = np.minimum(np.floor(p[:, :, 1].max(1) - .5).astype(np.int64), height - 1)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    a, bb, c = p[:, 0], p[:, 1], p[:, 2]
    area = (bb[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (bb[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    ok = (bw > 0) & (bh > 0) & (np.abs(area) > 1e-12)
    size_px = np.maximum(bw, bh)
    pix_all, z_all, t_all, w_all = [], [], [], []
    lo = 0
    for S in BUCKETS:
        sel = np.where(ok & (size_px > lo) & (size_px <= S))[0]
        lo = S
        if not len(sel):
            continue
        if S > 256:                                        # huge triangles: one at a time
            for t in sel:
                ys, xs = np.mgrid[y0[t]:y1[t] + 1, x0[t]:x1[t] + 1]
                cx, cy = xs.ravel() + .5, ys.ravel() + .5
                A_, B_, C_ = a[t], bb[t], c[t]
                w0 = ((B_[0] - cx) * (C_[1] - cy) - (B_[1] - cy) * (C_[0] - cx)) / area[t]
                w1 = ((C_[0] - cx) * (A_[1] - cy) - (C_[1] - cy) * (A_[0] - cx)) / area[t]
                w2 = 1 - w0 - w1
                inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
                if inside.any():
                    ww = np.stack([w0[inside], w1[inside], w2[inside]], 1)
                    pix_all.append(ys.ravel()[inside] * width + xs.ravel()[inside])
                    z_all.append(ww @ z[t]); t_all.append(np.full(int(inside.sum()), t)); w_all.append(ww)
            continue
        oy, ox = np.divmod(np.arange(S * S), S)
        px, py = x0[sel, None] + ox[None], y0[sel, None] + oy[None]
        inb = (ox[None] < bw[sel, None]) & (oy[None] < bh[sel, None])
        cx, cy = px + .5, py + .5
        A, B, C, ar = a[sel], bb[sel], c[sel], area[sel][:, None]
        w0 = ((B[:, 0, None] - cx) * (C[:, 1, None] - cy) - (B[:, 1, None] - cy) * (C[:, 0, None] - cx)) / ar
        w1 = ((C[:, 0, None] - cx) * (A[:, 1, None] - cy) - (C[:, 1, None] - cy) * (A[:, 0, None] - cx)) / ar
        w2 = 1 - w0 - w1
        inside = inb & (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        r, q = np.nonzero(inside)
        if not len(r):
            continue
        t = sel[r]
        ww = np.stack([w0[r, q], w1[r, q], w2[r, q]], 1)
        pix_all.append(py[r, q] * width + px[r, q]); z_all.append((ww * z[t]).sum(1)); t_all.append(t); w_all.append(ww)
    tri = np.full(width * height, -1, np.int64)
    bary = np.zeros((width * height, 3))
    zbuf = np.full(width * height, np.inf)
    if pix_all:
        P, Z = np.concatenate(pix_all), np.concatenate(z_all)
        T, WW = np.concatenate(t_all), np.concatenate(w_all)
        order = np.lexsort((Z, P))
        P = P[order]
        first = np.r_[True, P[1:] != P[:-1]]
        idx = order[first]
        tri[P[first]], bary[P[first]], zbuf[P[first]] = T[idx], WW[idx], Z[idx]
    return tri.reshape(height, width), bary.reshape(height, width, 3), zbuf.reshape(height, width)


def vertex_normals(X, tris):
    fn = np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]])
    n = np.zeros_like(X)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def render(X, cam, mesh, textures, background=174, tri_colour=None):
    """Textured Lambert render of posed vertices X (armature space) through a Camera. tri_colour: (triangles, 3) BGR
    painted over the texture where finite (the limbs of a painted start frame)."""
    w, h = cam.size
    tri, bary, _ = rasterize(cam.project(X), cam.depth(X), mesh['tris'], (w, h))
    valid = tri >= 0
    img = np.full((h, w, 3), background, np.float32)
    if not valid.any():
        return img.astype(np.uint8), valid
    t = tri[valid]
    uv = (bary[valid][:, :, None] * mesh['tri_uv'][t]).sum(1)
    col = np.zeros((len(t), 3), np.float32)
    mats = mesh['tri_material'][t]
    for m, tex in enumerate(textures):
        sel = mats == m
        if not sel.any():
            continue
        if isinstance(tex, np.ndarray) and tex.ndim == 3:
            th, tw = tex.shape[:2]
            u = np.clip((uv[sel, 0] % 1.0) * tw, 0, tw - 1).astype(int)
            v = np.clip((1 - uv[sel, 1] % 1.0) * th, 0, th - 1).astype(int)
            col[sel] = tex[v, u]
        else:
            col[sel] = tex
    if tri_colour is not None:
        paint = tri_colour[t]
        on = np.isfinite(paint[:, 0])
        col[on] = paint[on]
    vn = vertex_normals(X, mesh['tris'])
    n = (bary[valid][:, :, None] * vn[mesh['tris'][t]]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    view = cam.fwd @ cam.Ma                       # viewing direction in armature space
    n *= np.where((n @ view) > 0, -1, 1)[:, None]
    shade = np.full(len(n), .45)
    for d, k in LIGHTS:
        d = np.array(d, float)
        shade += k * np.clip(n @ (d / np.linalg.norm(d)), 0, None)
    img[valid] = np.clip(col * shade[:, None], 0, 255)
    return img.astype(np.uint8), valid


def load_textures(rig_dir, materials):
    out = []
    for m in materials:
        if 'texture' in m:
            img = cv2.imread(str(rig_dir / m['texture']), cv2.IMREAD_COLOR)
            out.append(img if img is not None else np.array([150, 150, 150], np.float32))
        else:
            r, g, b = [min(255.0, 255.0 * c ** (1 / 2.2)) for c in m.get('colour', (.6, .6, .6))]
            out.append(np.array([b, g, r], np.float32))
    return out
