"""The rig exported from Blender (blender/export_rig.py): forward kinematics, linear blend skinning and analytic
Jacobians in plain numpy, plus props (rigid pieces such as a log or a stone).

Parameters p of one frame (Rig.n numbers), all in the armature's own space:
  p[0:3]          offset of the root bone's head from its rest position
  p[3:3+C]        angles of the fitted bones, each in its own rest frame (Blender: Y runs along the bone, so the
                  middle angle is the twist): basis rotation = Rx(a) Ry(b) Rz(c); the ROOT's three are a rotation
                  vector instead (basis = exp(w)): the whole body may turn any way (roll onto its back, spin round,
                  somersault) with no axis limit and no gimbal lock half way (Euler angles lock at 90 degrees on
                  their middle axis, which is exactly a creature lying on its side)
  p[3+C:]         per prop: offset (3) and rotation Rx Ry Rz (3) about its centre at the start, in armature space
Bones that are not fitted keep the basis they have in the start pose. Skinning uses the four strongest influences
per vertex, normalised, as Blender and glTF do. Prop vertices carry pseudo bone ids after the real bones.
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation

ROOT_NAMES = ('hips', 'pelvis', 'mixamorig:hips', 'root_hips')


def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_so3(w):
    """Rotation matrix of the rotation vector w (Rodrigues)."""
    th = float(np.linalg.norm(w))
    K = hat(w)
    if th < 1e-8:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def right_jacobian(w):
    """exp(w + d) = exp(w) exp(Jr(w) d) for a small d: the local axes a change of each component turns about."""
    th = float(np.linalg.norm(w))
    K = hat(w)
    if th < 1e-6:
        return np.eye(3) - .5 * K + K @ K / 6.0
    return np.eye(3) - (1 - np.cos(th)) / th ** 2 * K + (th - np.sin(th)) / th ** 3 * K @ K


class Rig:
    def __init__(self, data, fitted, root=None):
        bones = data['bones']
        self.names = [b['name'] for b in bones]
        self.nb = len(self.names)
        self.parents = np.array([-1 if b['parent'] is None else self.names.index(b['parent']) for b in bones])
        if not all(self.parents[j] < j for j in range(self.nb)):
            raise SystemExit('rig.json: bones must be listed parents first (export_rig.py does this)')
        rest = np.array([b['rest'] for b in bones], float)
        self.rest_rot, self.rest_head = rest[:, :3, :3], rest[:, :3, 3]
        self.length = np.array([b['length'] for b in bones])
        rl = np.array([np.linalg.inv(rest[self.parents[j]]) @ rest[j] if self.parents[j] >= 0 else rest[j]
                       for j in range(self.nb)])
        self.Rrl, self.trl = rl[:, :3, :3], rl[:, :3, 3]
        if root is None:
            lower = [n.lower() for n in self.names]
            root = next((self.names[lower.index(r)] for r in ROOT_NAMES if r in lower), self.names[0])
        self.root = self.names.index(root)
        start = data.get('start_pose', {})
        basis = {n: np.array(start.get(n, np.eye(4)), float) for n in self.names}
        self.fixed = np.array([basis[n][:3, :3] for n in self.names])      # start-pose rotation of every bone
        self.root_loc0 = basis[self.names[self.root]][:3, 3]
        self.channels = [(self.names.index(b), a) for b in fitted for a in (0, 1, 2)]
        self.nc = len(self.channels)
        self.chan_of = {}
        for c, (j, a) in enumerate(self.channels):
            self.chan_of.setdefault(j, []).append((a, 3 + c))
        self.props = data.get('props', [])
        self.prop0 = 3 + self.nc
        self.prop_centre = np.array([pr['centre'] for pr in self.props], float).reshape(-1, 3)
        self.n = self.prop0 + 6 * len(self.props)
        self.desc = np.eye(self.nb, dtype=bool)
        for j in range(self.nb - 1, 0, -1):
            if self.parents[j] >= 0:
                self.desc[self.parents[j]] |= self.desc[j]
        # The root's start-pose translation is part of p0 (an offset in armature space), not a fixed basis.
        self.root_loc0_world = None

    def index(self, bone, axis):
        return 3 + self.channels.index((self.names.index(bone), axis))

    # ----- the start pose ---------------------------------------------------------------------------------
    def start_params(self):
        """Parameters of the start pose (the pose the start frame was rendered in)."""
        p = np.zeros(self.n)
        for j, chans in self.chan_of.items():
            if j == self.root:
                ang = Rotation.from_matrix(self.fixed[j]).as_rotvec()      # the root: a rotation vector
            else:
                ang = Rotation.from_matrix(self.fixed[j]).as_euler('XYZ')  # R = Rx(a) Ry(b) Rz(c)
            for a, k in chans:
                p[k] = ang[a]
        G, _, _ = self.fk(p, root_offset=False)
        par = self.parents[self.root]
        frame = (G[par] if par >= 0 else np.eye(3)) @ self.Rrl[self.root]
        p[:3] = frame @ self.root_loc0
        return p

    def params_from_basis(self, basis, names, heads=None):
        """Parameters of a pose given as Blender pose-bone bases (one 4x4 per bone, `names` their order; another
        tool's animation applied to the same armature). Every fitted bone gets the world rotation that pose gives
        it, also when bones we do not fit (a "Root" above the hips) moved in it; the root's offset puts its head
        where `heads` (armature space, same order) has it, else where the basis translations put it."""
        B = {n: np.asarray(basis[i], float) for i, n in enumerate(names)}
        Gm = np.zeros((self.nb, 3, 3))
        for j in range(self.nb):
            par = self.parents[j]
            R = B.get(self.names[j], np.eye(4))[:3, :3]
            Gm[j] = (Gm[par] if par >= 0 else np.eye(3)) @ self.Rrl[j] @ R
        p = np.zeros(self.n)
        G = np.zeros((self.nb, 3, 3))
        for j in range(self.nb):
            par = self.parents[j]
            base = (G[par] if par >= 0 else np.eye(3)) @ self.Rrl[j]
            if j in self.chan_of:
                local = base.T @ Gm[j]
                ang = (Rotation.from_matrix(local).as_rotvec() if j == self.root
                       else Rotation.from_matrix(local).as_euler('XYZ'))
                for a, k in self.chan_of[j]:
                    p[k] = ang[a]
                G[j] = Gm[j]
            else:
                G[j] = base @ self.fixed[j]
        P0, _ = self.bone_points(np.concatenate([np.zeros(3), p[3:]]))
        if heads is not None:
            p[:3] = np.asarray(heads, float)[list(names).index(self.names[self.root])] - P0[self.root]
        else:
            par = self.parents[self.root]
            frame = (G[par] if par >= 0 else np.eye(3)) @ self.Rrl[self.root]
            p[:3] = frame @ B.get(self.names[self.root], np.eye(4))[:3, 3]
        return p

    # ----- kinematics ---------------------------------------------------------------------------------------
    def fk(self, p, root_offset=True):
        """World (armature-space) rotation G and head P of every bone; the rotation axes for the Jacobian."""
        key = (np.asarray(p, float).tobytes(), root_offset)
        if getattr(self, '_fk_key', None) == key:
            return self._fk
        G = np.zeros((self.nb, 3, 3))
        P = np.zeros((self.nb, 3))
        axes = []
        for j in range(self.nb):
            par = self.parents[j]
            if par < 0:
                Gj = self.Rrl[j].copy()
                P[j] = self.trl[j]
            else:
                Gj = G[par] @ self.Rrl[j]
                P[j] = P[par] + G[par] @ self.trl[j]
            if j == self.root and root_offset:
                P[j] = P[j] + p[:3]
            if j in self.chan_of and j == self.root:
                # rotation vector: a change of component k turns the bone about the world axis G Jr(w)[:, k]
                chans = self.chan_of[j]
                w = np.array([p[k] for _, k in sorted(chans)])
                Gj = Gj @ exp_so3(w)
                Ja = Gj @ right_jacobian(w)
                for a, k in chans:
                    axes.append((j, Ja[:, a].copy(), k))
            elif j in self.chan_of:
                for a, k in self.chan_of[j]:
                    axes.append((j, Gj[:, a].copy(), k))
                    Gj = Gj @ rot(a, p[k])
            else:
                Gj = Gj @ self.fixed[j]
            G[j] = Gj
        self._fk_key, self._fk = key, (G, P, axes)
        return self._fk

    def prop_rotation(self, p, k):
        a, b, c = p[self.prop0 + 6 * k + 3:self.prop0 + 6 * k + 6]
        return rot(0, a) @ rot(1, b) @ rot(2, c)

    def skin(self, p, rest, idx, wts, jac=False):
        is_prop = idx[:, 0] >= self.nb
        out = np.zeros((len(rest), 3))
        J = np.zeros((len(rest), 3, self.n)) if jac else None
        body = ~is_prop
        if body.any():
            G, P, axes = self.fk(p)
            bi, bw, br = idx[body], wts[body], rest[body]
            Mk = np.einsum('nij,nkj->nik', G, self.rest_rot)
            tk = P - np.einsum('nij,nj->ni', Mk, self.rest_head)
            q = np.einsum('vkij,vj->vki', Mk[bi], br) + tk[bi]
            out[body] = (q * bw[:, :, None]).sum(1)
            if jac:
                Jb = np.zeros((len(br), 3, self.n))
                root_share = (self.desc[self.root, bi] * bw).sum(1)
                Jb[:, :, :3] = np.eye(3)[None] * root_share[:, None, None]
                if axes:
                    # every rotation axis at once: the share of each vertex its bone's subtree moves, the moment arm
                    aj = np.array([a[0] for a in axes])
                    axis = np.array([a[1] for a in axes])
                    ak = np.array([a[2] for a in axes])
                    infl = self.desc[aj][:, bi] * bw[None]                       # (axes, vertices, weights)
                    part = np.einsum('ank,nkd->and', infl, q) - infl.sum(2)[:, :, None] * P[aj][:, None, :]
                    # a bone above the root (a Mixamo-style "Root"): the root's offset is added in armature space
                    # after it, so it does not turn with it
                    above = (aj != self.root) & self.desc[aj, self.root]
                    if above.any():
                        part[above] -= root_share[None, :, None] * p[:3]
                    Jb[:, :, ak] += np.cross(axis[:, None, :], part).transpose(1, 2, 0)
                J[body] = Jb
        for k in np.unique(idx[is_prop, 0] - self.nb):
            sel = is_prop & (idx[:, 0] == self.nb + k)
            c0 = self.prop_centre[k]
            base = self.prop0 + 6 * k
            R = self.prop_rotation(p, k)
            d = (rest[sel] - c0) @ R.T
            out[sel] = d + c0 + p[base:base + 3]
            if jac:
                a, b = p[base + 3], p[base + 4]
                ax = [np.array([1.0, 0, 0]), rot(0, a) @ np.array([0, 1.0, 0]), rot(0, a) @ rot(1, b) @ np.array([0, 0, 1.0])]
                Jp = np.zeros((int(sel.sum()), 3, self.n))
                Jp[:, :, base:base + 3] = np.eye(3)
                for i in range(3):
                    Jp[:, :, base + 3 + i] = np.cross(ax[i], d)
                J[sel] = Jp
        return (out, J) if jac else out

    def bone_points(self, p):
        G, P, _ = self.fk(p)
        return P, P + G[:, :, 1] * self.length[:, None]

    def pose_basis(self, p):
        """Blender pose-bone rotation (quaternion w, x, y, z) of every bone, the root's location (bone space),
        and every prop's world matrix (armature space) for this frame."""
        quats = np.zeros((self.nb, 4))
        for j in range(self.nb):
            R = np.eye(3)
            if j in self.chan_of and j == self.root:
                R = exp_so3(np.array([p[k] for _, k in sorted(self.chan_of[j])]))
            elif j in self.chan_of:
                for a, k in self.chan_of[j]:
                    R = R @ rot(a, p[k])
            else:
                R = self.fixed[j]
            w, x, y, z = Rotation.from_matrix(R).as_quat(scalar_first=True)
            quats[j] = [w, x, y, z]
        G, _, _ = self.fk(p)
        par = self.parents[self.root]
        frame = (G[par] if par >= 0 else np.eye(3)) @ self.Rrl[self.root]
        loc = frame.T @ p[:3]
        props = []
        for k in range(len(self.props)):
            M = np.eye(4)
            R = self.prop_rotation(p, k)
            c0 = self.prop_centre[k]
            M[:3, :3] = R
            M[:3, 3] = c0 + p[self.prop0 + 6 * k:self.prop0 + 6 * k + 3] - R @ c0
            props.append(M)
        return quats, loc, props
