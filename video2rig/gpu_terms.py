"""The solver's cost terms for many frames at once on the graphics card (PyTorch, CUDA): the same terms as
Problem.frame_terms (tracks, inside and covering the silhouette, floor, prior, depth), returned as each frame's
normal equations J'J, J'r and cost, which is all the Levenberg-Marquardt steps need. Skinning and its Jacobian for
every vertex of every frame in one go replace a Python loop over frames that took most of a solve.

Props (a log, a stone) are rigid bodies with their own offset and turn. Creatures with bend terms use the processor
path (Problem.frame_terms).
"""
import numpy as np
import torch

CHUNK = 12          # frames per pass on the card at most
CHUNK_BYTES = 2.5e8  # and fewer for big rigs: the moment arms of every axis for every vertex stay under this


class GpuTerms:
    def __init__(self, pb, dev):
        self.pb, self.dev = pb, dev
        rig = pb.rig
        f64 = dict(dtype=torch.float64, device=dev)
        self.f64 = f64
        T = lambda a, **k: torch.as_tensor(np.ascontiguousarray(a), **(k or f64))
        long = dict(dtype=torch.int64, device=dev)
        self.rest = T(pb.rest)
        body = pb.idx[:, 0] < rig.nb
        self.bi = T(np.where(body[:, None], pb.idx, 0), **long)      # prop vertices: filled in by their own transform
        self.bw = T(np.where(body[:, None], pb.wts, 0.0))
        self.prop_of = T(np.where(body, -1, pb.idx[:, 0] - rig.nb), **long)          # prop index per vertex, -1 body
        self.V = len(pb.rest)
        self.need_base = np.unique(np.concatenate([pb.tv.ravel(), pb.sil, pb.floor, pb.cbody, pb.axis_v, pb.tip_v,
                                                   pb.kp_v, pb.flat.ravel()]
                                                  + [pr['pv'] for pr in pb.props]))
        self.prop_c0 = T(rig.prop_centre) if len(rig.props) else None
        self.props = [dict(k=pr['k'], eq=T(pr['eq']), pv=T(pr['pv'], **long)) for pr in pb.props]
        self.cbody = T(pb.cbody, **long)
        self.rest_rot, self.rest_head = T(rig.rest_rot), T(rig.rest_head)
        self.desc = T(rig.desc.astype(np.float64))
        self.root_share = (self.desc[rig.root][self.bi] * self.bw).sum(1)              # (V,)
        self.infl = None                                                               # (axes, V, K), first fk
        # tracks
        self.tv, self.tbary = T(pb.tv, **long), T(pb.bary)
        self.u = T(pb.u)                                                               # (K, T, 2)
        self.rigid = T(pb.rigid)
        self.view_tracks = [np.where(pb.tview == vi)[0] for vi in range(len(pb.views))]
        nparts = len(pb.part_names)
        self.onehot = [torch.nn.functional.one_hot(T(pb.tpart[ids], **long), nparts).to(torch.float64)
                       for ids in self.view_tracks]
        self.tpart = [T(pb.tpart[ids], **long) for ids in self.view_tracks]
        self.sil, self.floor = T(pb.sil, **long), T(pb.floor, **long)
        self.up = T(pb.up)
        self.dout = [torch.as_tensor(np.stack(per), dtype=torch.float32, device=dev) for per in pb.dout]
        self.tris = T(pb.tris, **long)
        self.prior_w = T(pb.prior_w)
        self.base = T(pb.base[pb.act])
        self.p0 = T(pb.p0[pb.act])
        depth_row = np.zeros(pb.na)
        if all(c in pb.act for c in (0, 1, 2)) and pb.depth_axis is not None:
            depth_row[[list(pb.act).index(c) for c in (0, 1, 2)]] = pb.depth_axis
        self.depth_row = T(depth_row)
        self.cams = []
        for v in pb.views:
            cam = v['cam']
            if cam.type == 'ortho':
                self.cams.append(('ortho', T(cam.A), T(cam.b)))
            else:
                c = np.array([cam.w / 2, cam.h / 2]) + cam.shift + cam.offset
                self.cams.append(('persp', T(cam.Ma), T(cam.ta), T(cam.loc), T(cam.right), T(cam.up), T(cam.fwd),
                                  float(cam.f0 * cam.s), T(c)))

    @staticmethod
    def usable(pb):
        return not pb.bends

    # ----- pieces --------------------------------------------------------------------------------------------
    def project(self, vi, X):
        """Pixels (..., 2) of points X (..., 3) and d pixel / d point (..., 2, 3)."""
        cam = self.cams[vi]
        if cam[0] == 'ortho':
            A, b = cam[1], cam[2]
            return X @ A.T + b, A.expand(*X.shape[:-1], 2, 3)
        _, Ma, ta, loc, right, up, fwd, f, c = cam
        d = X @ Ma.T + ta - loc
        x, y, z = d @ right, d @ up, torch.clamp(d @ fwd, min=1e-6)
        pix = torch.stack([c[0] + f * x / z, c[1] - f * y / z], -1)
        Ju = f * (right / z[..., None] - (x / z ** 2)[..., None] * fwd)
        Jv = -f * (up / z[..., None] - (y / z ** 2)[..., None] * fwd)
        return pix, torch.stack([Ju, Jv], -2) @ Ma

    @staticmethod
    def bilinear(field, pix):
        """field (C, H, W) sampled at pixel positions pix (C, N, 2) like cv2.remap(INTER_LINEAR, BORDER_REPLICATE) at
        pix - 0.5: (C, N)."""
        C, H, W = field.shape
        x = (pix[..., 0] - .5).float()
        y = (pix[..., 1] - .5).float()
        x0, y0 = torch.floor(x), torch.floor(y)
        fx, fy = x - x0, y - y0
        x0, y0 = x0.long(), y0.long()
        flat = field.reshape(C, -1)

        def at(yy, xx):
            idx = yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)
            return torch.gather(flat, 1, idx)
        v = (at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy)
             + at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy)
        return v.double()

    def skin(self, xs, jac, need):
        """Posed vertices need (C, N, 3) and d vertex / d parameter (C, N, 3, na) for the frames' parameters xs."""
        pb, rig = self.pb, self.pb.rig
        Gs, Ps, axes_all = [], [], []
        for x in xs:
            G, P, axes = rig.fk(pb.full(x))
            Gs.append(G)
            Ps.append(P)
            axes_all.append(np.array([a[1] for a in axes]))
        if self.infl is None:
            aj = np.array([a[0] for a in axes])
            self.ak = torch.as_tensor(np.array([a[2] for a in axes]), dtype=torch.int64, device=self.dev)
            self.aj = torch.as_tensor(aj, dtype=torch.int64, device=self.dev)
            self.infl = self.desc[self.aj][:, self.bi] * self.bw[None]                # (A, V, K)
            self.above = torch.as_tensor((aj != rig.root) & rig.desc[aj, rig.root], device=self.dev)
        G = torch.as_tensor(np.stack(Gs), **self.f64)
        P = torch.as_tensor(np.stack(Ps), **self.f64)
        Mk = torch.einsum('cnij,nkj->cnik', G, self.rest_rot)
        tk = P - torch.einsum('cnij,nj->cni', Mk, self.rest_head)
        bi, bw, rest, prop_of = self.bi[need], self.bw[need], self.rest[need], self.prop_of[need]
        q = torch.einsum('cvkij,vj->cvki', Mk[:, bi], rest) + tk[:, bi]                  # (C, N, K, 3)
        X = (q * bw[None, :, :, None]).sum(2)
        if not jac:
            self.prop_skin(xs, X, None, rest, prop_of)
            return X, None
        C, V = X.shape[:2]
        root_share = self.root_share[need]
        J = torch.zeros((C, V, 3, rig.n), **self.f64)
        J[:, :, :, :3] = torch.eye(3, **self.f64)[None, None] * root_share[None, :, None, None]
        infl = self.infl[:, need]
        part = torch.einsum('avk,cvkd->cavd', infl, q) - infl.sum(2)[None, :, :, None] * P[:, self.aj][:, :, None, :]
        if bool(self.above.any()):
            off = torch.as_tensor(np.stack([pb.full(x)[:3] for x in xs]), **self.f64)
            part[:, self.above] -= root_share[None, None, :, None] * off[:, None, None, :]
        axis = torch.as_tensor(np.stack(axes_all), **self.f64)                         # (C, A, 3)
        cr = torch.cross(axis[:, :, None, :].expand_as(part), part, dim=-1)            # (C, A, N, 3)
        J[:, :, :, self.ak] += cr.permute(0, 2, 3, 1)
        self.prop_skin(xs, X, J, rest, prop_of)
        act = torch.as_tensor(pb.act, dtype=torch.int64, device=self.dev)
        return X, J[..., act]

    def pose_all(self, x):
        """Every vertex of the mesh posed by one frame's parameters x (na,), as numpy (V, 3)."""
        if not hasattr(self, 'all_rows'):
            self.all_rows = torch.arange(self.V, device=self.dev)
        X, _ = self.skin(np.asarray(x)[None], False, self.all_rows)
        return X[0].cpu().numpy()

    def prop_turns(self, xs, k):
        """Each frame's turn R of prop k and its three rotation axes (as rig.prop_rotation and prop_terms)."""
        rig = self.pb.rig
        Rs, axs = [], []
        for x in xs:
            p = self.pb.full(x)
            base = rig.prop0 + 6 * k
            a, b = p[base + 3], p[base + 4]
            Rs.append(rig.prop_rotation(p, k))
            rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
            ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
            axs.append([[1.0, 0, 0], rx @ [0, 1.0, 0], rx @ ry @ [0, 0, 1.0]])
        return torch.as_tensor(np.stack(Rs), **self.f64), torch.as_tensor(np.array(axs), **self.f64)

    def prop_skin(self, xs, X, J, rest, prop_of):
        """A prop's vertices move rigidly with its own offset and turn (rig.skin)."""
        rig = self.pb.rig
        for k in range(len(rig.props)):
            rows = torch.nonzero(prop_of == k).squeeze(1)
            if not len(rows):
                continue
            base = rig.prop0 + 6 * k
            R, ax = self.prop_turns(xs, k)
            tt = torch.as_tensor(np.stack([self.pb.full(x)[base:base + 3] for x in xs]), **self.f64)
            d = torch.einsum('vj,cij->cvi', rest[rows] - self.prop_c0[k], R)
            X[:, rows] = d + self.prop_c0[k] + tt[:, None]
            if J is not None:
                J[:, rows] = 0.0
                J[:, rows, :, base:base + 3] = torch.eye(3, **self.f64)
                for i in range(3):
                    J[:, rows, :, base + 3 + i] = torch.cross(ax[:, None, i].expand_as(d), d, dim=-1)

    # ----- terms ---------------------------------------------------------------------------------------------
    def frames(self, frames, xs, robust=None, want_jac=True):
        """Normal equations of the given frames: H (C, na, na), g (C, na), cost (C,) as numpy; cost only without
        the Jacobian. robust: per-track weights (C, K) or None."""
        out_H, out_g, out_c = [], [], []
        axes = len(self.pb.rig.channels)
        step = CHUNK if not want_jac else max(1, min(CHUNK, int(CHUNK_BYTES / (axes * len(self.need_base) * 24 * 1.3))))
        for k in range(0, len(frames), step):
            fr = list(frames[k:k + step])
            rb = None if robust is None else robust[k:k + step]
            H, g, c = self.chunk(fr, np.asarray(xs[k:k + step]), rb, want_jac)
            out_c.append(c)
            if want_jac:
                out_H.append(H)
                out_g.append(g)
        cost = torch.cat(out_c).cpu().numpy()
        if not want_jac:
            return cost
        return torch.cat(out_H).cpu().numpy(), torch.cat(out_g).cpu().numpy(), cost

    def chunk(self, fr, xs, robust, want_jac):
        pb, w = self.pb, self.pb.w
        C, na = len(fr), pb.na
        covers = [[pb.cover[t][vi] for t in fr] for vi in range(len(pb.views))]
        extra = [self.pb.tris[cv['tri']].ravel() for per in covers for cv in per if cv is not None and len(cv['tri'])]
        need = np.unique(np.concatenate([self.need_base] + extra))
        loc_np = np.full(self.V, -1, np.int64)
        loc_np[need] = np.arange(len(need))
        loc = torch.as_tensor(loc_np, device=self.dev)
        X, J = self.skin(xs, want_jac, torch.as_tensor(need, device=self.dev))
        self.loc = loc
        cost = torch.zeros(C, **self.f64)
        H = torch.zeros((C, na, na), **self.f64) if want_jac else None
        g = torch.zeros((C, na), **self.f64) if want_jac else None
        ci = torch.arange(C, device=self.dev)

        def add(r, Jr=None):
            nonlocal cost
            cost = cost + (r * r).sum(1)
            if want_jac and Jr is not None:
                H.add_(torch.einsum('crn,crm->cnm', Jr, Jr))
                g.add_(torch.einsum('crn,cr->cn', Jr, r))
        frt = torch.as_tensor(fr, dtype=torch.int64, device=self.dev)
        rel = torch.as_tensor(pb.rel, **self.f64)
        for vi in range(len(pb.views)):
            # tracks: each body part normalised on its own (per view), weighted by visibility, agreement, rigidity
            ids = self.view_tracks[vi]
            if len(ids):
                idt = torch.as_tensor(ids, dtype=torch.int64, device=self.dev)
                vis = torch.as_tensor(pb.vis[np.ix_(ids, fr)].T, **self.f64)           # (C, Kv)
                seen = vis >= w['visible_min']
                tv, bary = loc[self.tv[idt]], self.tbary[idt]
                pts = torch.einsum('km,ckmd->ckd', bary, X[:, tv])
                pix, dP = self.project(vi, pts)
                e = pix - self.u[idt][:, frt].permute(1, 0, 2)
                ww = vis * (rel[idt] * self.rigid[idt])[None]
                if robust is not None:
                    ww = ww * torch.as_tensor(np.asarray(robust)[:, ids], **self.f64)
                ww = torch.where(seen, ww, torch.zeros_like(ww))
                cnt = seen.double() @ self.onehot[vi]                                    # (C, parts)
                norm = torch.clamp(cnt, min=float(w['track_min_points']))
                s = torch.sqrt(ww * w['w_track'] / torch.gather(norm, 1, self.tpart[vi][None].expand(C, -1)))
                r = (e * s[..., None]).reshape(C, -1)
                Jr = None
                if want_jac:
                    Jp = torch.einsum('km,ckmdn->ckdn', bary, J[:, tv])
                    Jr = (torch.einsum('ckad,ckdn->ckan', dP, Jp) * s[..., None, None]).reshape(C, -1, na)
                add(r, Jr)
            # inside the silhouette
            sil = loc[self.sil]
            pts = X[:, sil]
            pix, dP = self.project(vi, pts)
            field = self.dout[vi][frt]
            d = self.bilinear(field, pix)
            s = float(np.sqrt(w['w_inside'] / len(self.sil)))
            Jr = None
            if want_jac:
                one = torch.tensor([1.0, 0.0], **self.f64)
                gx = (self.bilinear(field, pix + one) - self.bilinear(field, pix - one)) / 2
                gy = (self.bilinear(field, pix + one.flip(0)) - self.bilinear(field, pix - one.flip(0))) / 2
                g3 = gx[..., None] * dP[..., 0, :] + gy[..., None] * dP[..., 1, :]
                Jr = torch.einsum('csd,csdn->csn', g3, J[:, sil]) * s
            add(d * s, Jr)
            # video silhouette the model leaves uncovered: the nearest model outline point pulled there
            M = max((len(cv['target']) for cv in covers[vi] if cv is not None), default=0)
            if M:
                tri = np.zeros((C, M), np.int64)
                bc = np.zeros((C, M, 3))
                tg = np.zeros((C, M, 2))
                wt = np.zeros((C, M))                       # each match's weight (0: padding)
                cc = np.full((C, M), float(w['cover_cauchy_px']))   # and its robust scale
                for c_, cv in enumerate(covers[vi]):
                    if cv is None or not len(cv['target']):
                        continue
                    m_ = len(cv['target'])
                    tri[c_, :m_], bc[c_, :m_], tg[c_, :m_] = cv['tri'], cv['bary'], cv['target']
                    wt[c_, :m_] = cv['w'] if 'w' in cv else w['w_cover'] / max(m_, w['cover_min_points'])
                    if 'c' in cv:
                        cc[c_, :m_] = cv['c']
                vt = loc[self.tris[torch.as_tensor(tri, dtype=torch.int64, device=self.dev)]]   # (C, M, 3)
                bc = torch.as_tensor(bc, **self.f64)
                pts = torch.einsum('cmk,cmkd->cmd', bc, X[ci[:, None, None], vt])
                pix, dPc = self.project(vi, pts)
                e = pix - torch.as_tensor(tg, **self.f64)
                rw = 1 / (1 + (torch.linalg.norm(e, dim=-1) / torch.as_tensor(cc, **self.f64)) ** 2)
                s = torch.sqrt(rw * torch.as_tensor(wt, **self.f64))
                Jr = None
                if want_jac:
                    Jp = torch.einsum('cmk,cmkdn->cmdn', bc, J[ci[:, None, None], vt])
                    Jr = (torch.einsum('cmad,cmdn->cman', dPc, Jp) * s[..., None, None]).reshape(C, -1, na)
                add((e * s[..., None]).reshape(C, -1), Jr)
        for pr in self.props:
            add(*self.prop_terms(pr, xs, X, J, want_jac))
        # a foot on the floor lies as flat as at the start
        if len(pb.flat):
            fa = loc[torch.as_tensor(pb.flat[:, 0], device=self.dev)]
            fb = loc[torch.as_tensor(pb.flat[:, 1], device=self.dev)]
            za, zb = X[:, fa] @ self.up, X[:, fb] @ self.up
            on = (torch.minimum(za, zb) - pb.floor_level < w['contact'] * pb.height).double()
            k_ = w['w_flat'] / pb.unit
            drop = torch.as_tensor(pb.flat_drop, **self.f64)
            add(((zb - za) - drop) * on * k_, torch.einsum('d,cfdn->cfn', self.up, J[:, fb] - J[:, fa])
                * (on * k_)[..., None] if want_jac else None)
        # floor
        floor = loc[self.floor]
        z = X[:, floor] @ self.up - pb.floor_level
        below = (z < 0).double()
        k_ = w['w_floor'] / pb.unit
        add(z * below * k_, torch.einsum('d,cfdn->cfn', self.up, J[:, floor]) * (below * k_)[..., None]
            if want_jac else None)
        # prior and depth: the same rows in every frame
        xt = torch.as_tensor(xs, **self.f64)
        r = (xt - self.base) * self.prior_w
        cost = cost + (r * r).sum(1)
        dr = self.depth_row * (w['w_depth'] / pb.unit)
        rd = (xt - self.p0) @ dr
        cost = cost + rd * rd
        if want_jac:
            H.add_(torch.diag(self.prior_w ** 2)[None] + torch.outer(dr, dr)[None])
            g.add_(r * self.prior_w + rd[:, None] * dr[None])
        if pb.gait_w is not None:                          # a hidden leg pulled into its twin's cycle
            gw = torch.as_tensor(pb.gait_w[fr], **self.f64)
            rg = (xt - torch.as_tensor(pb.gait_target[fr], **self.f64)) * gw
            cost = cost + (rg * rg).sum(1)
            if want_jac:
                H.add_(torch.diag_embed(gw ** 2))
                g.add_(rg * gw)
        return H, g, cost

    def prop_terms(self, pr, xs, X, J, want_jac):
        """As Problem.prop_terms for every frame: the body samples inside the prop (convex hull, rest frame) pushed
        out, and when the prop is off the floor and nothing is inside, the nearest sample pulled onto its surface."""
        pb, w, rig = self.pb, self.pb.w, self.pb.rig
        k = pr['k']
        base = rig.prop0 + 6 * k
        R, ax = self.prop_turns(xs, k)
        c0 = self.prop_c0[k]
        tt = torch.as_tensor(np.stack([pb.full(x)[base:base + 3] for x in xs]), **self.f64)
        cbody = self.loc[self.cbody]
        xb = X[:, cbody]                                                               # (C, B, 3)
        rel = xb - c0 - tt[:, None]
        eq = pr['eq']
        sp_ = (torch.einsum('cbi,cij->cbj', rel, R) + c0) @ eq[:, :3].T + eq[:, 3]
        sd, j = sp_.max(2)
        m = torch.einsum('cbi,cji->cbj', eq[j, :3], R)                                 # active plane's normal, world
        lowest = (X[:, self.loc[pr['pv']]] @ self.up).amin(1) - pb.floor_level
        lifted = lowest > w['prop_lift'] * pb.height
        inside = sd < 0
        n_in = inside.sum(1)
        s_in, s_t = w['w_prop_inside'] / pb.unit, w['w_prop_touch'] / pb.unit
        scale_in = inside.double() * s_in / torch.sqrt(torch.clamp(n_in, min=1).double())[:, None]
        touch = sd.argmin(1)
        ci = torch.arange(len(xs), device=self.dev)
        gap = torch.clamp(sd[ci, touch] - w['prop_gap'] * pb.height, min=0.0)
        gap = torch.where(lifted & (n_in == 0), gap, torch.zeros_like(gap))
        r = torch.cat([sd * scale_in, (gap * s_t)[:, None]], 1)
        if not want_jac:
            return r, None
        grad = torch.einsum('cbd,cbdn->cbn', m, J[:, cbody])                           # through the body
        act = list(pb.act)
        for q in range(3):
            if base + q in act:
                grad[:, :, act.index(base + q)] -= m[:, :, q]
            if base + 3 + q in act:
                grad[:, :, act.index(base + 3 + q)] += torch.einsum(
                    'cd,cbd->cb', ax[:, q], torch.cross(m, rel, dim=-1))
        Jr = torch.cat([grad * scale_in[..., None], (grad[ci, touch] * (s_t * (gap > 0).double())[:, None])[:, None]], 1)
        return r, Jr
