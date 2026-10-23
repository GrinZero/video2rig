"""Fit the rig to a whole clip at once.

Unknowns: every frame's pose (the creature's own parameters; only the active ones move). Cost, in pixel units:
  tracks     each tracked mesh point projects onto its track (robust: Cauchy; weighted by the tracker's
             visibility and by how well the track agrees with the rig over the clip); every body part (body, each
             leg) is normalised on its own, so a half-hidden far leg still counts as a whole leg
  inside     model points project inside the video silhouette (distance outside it)
  prior      roll / yaw / abduction and depth stay near the base pose (one camera barely sees them)
  floor      no point under the floor; bends: knees and heels keep their bend direction
  props      no body point inside a prop (its convex hull); a prop off the floor rests on the body (hands, shoulder):
             one camera does not see a prop's depth, and a lifted log or stone cannot float beside the body
  smooth     joint accelerations in content time (Seedance's uneven frame clock), per channel group; a part that has
             lost most of its tracked points in a frame (hidden by a prop or the body) is held up to 5x more firmly
             to its neighbouring frames, so it is filled in smoothly instead of bending wherever the outline allows
First a warm start frame by frame (each from the one before), then Levenberg-Marquardt over all frames together,
re-weighting the robust terms between rounds.
"""
import json
import time

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from scipy.spatial.transform import Rotation

from .camera import depth as view_depth
from .camera import project
from .creature import FINGER
from .raster import device
from .rig import exp_so3

DEFAULTS = {
    'w_track': 400.0,        # per part: mean squared track error (px^2) x this
    'track_min_points': 30,  # a part with fewer seen points is normalised as if it had this many
    'cauchy_px': 5.0,
    'robust_start': 8.0,     # first round's robust scale x this, down to x1 by the last: a limb the first pass left
                             # behind is pulled back by its own tracks before they count as outliers (no lag, no pop)
    'visible_min': .3,
    'w_inside': 150.0,       # mean squared distance outside the silhouette x this
    'inside_dilate_px': 2,
    'inside_cap_px': 40.0,
    'sil_stride': 3,           # every n-th vertex must project inside the silhouette
    'w_cover': 150.0,        # mean squared distance of uncovered silhouette to the model outline x this
    'cover_cauchy_px': 8.0,
    'cover_min_points': 40,
    'cover_min_area': 30,     # px of uncovered silhouette before the term turns on in a frame
    'cover_reach_px': 60.0,
    'w_parts': 150.0,        # each limb into its own outline (SAM 2, parts.py) and its outline onto it; 0 = off
    'colour_parts': True,    # a clip from a painted start frame: each limb's outline read from its colour (colours.py)
                             # instead of SAM 2; the colours measure every limb, so mirror, gait, tip and keypoints go
    'parts_cauchy_px': 8.0,  # robust scale of the limb outline matches (px): a limb further off than a few times this
                             # is barely pulled back to its own outline
    'w_axis': 150.0,         # points along each lower-limb bone onto the centre line of its limb's outline
    'w_tip': 200.0,          # each limb's last bone ends at its outline's far end (a paw, a hoof, a hand)
    'tip_cauchy_px': 120.0,  # and pulls from far: a leg left folded when the blur took its tracks still comes out
    'w_keypoints': 100.0,    # an animal pose model's elbow, knee or paw (keypoints.py) on the rig's joint, x its score
    'keypoint_cauchy_px': 25.0,
    'w_flat': 2000.0,        # px-equivalent per metre: a foot on the floor lies as flat as in the start pose
    'contact': .03,          # a foot is on the floor within this share of the creature's height
    'mirror_legs': True,     # a poorly tracked leg bends within the range its well tracked mirror twin used
    'mirror_margin_deg': 15.0,
    'w_gait': 60.0,          # px-equivalent per rad: a hidden leg follows its twin's cycle, shifted (0 = off)
    'gait_min_corr': .5,     # the twin's motion repeats at least this well (autocorrelation) to count as a gait
    'pin_start_deg': 0.0,    # frame 0 (the start render, a known pose) moves at most this much; 0 = free
    'w_guide': 60.0,         # px-equivalent per radian toward a guide (solve(guide=...), --guide): 0 = start only
    'guide_offaxis': 10.0,   # twist channels (about the bone) follow the guide this many times more firmly than bends
    'guide_side': 2.0,       # sideways channels: firmer than bends, but free enough for a paw to come in under the belly
    'despike_deg': 6.0,      # a joint that jumps away for one frame and comes back (blur, dust) by more than this is set
                             # to the mean of its neighbours, as mocap is cleaned; 0 = off
    'lowpass_hz': 8.0,       # the finished motion through a zero-phase low-pass (Butterworth, 2nd order); 0 = off
    'guide_root': 0.0,       # the body's own turn follows the guide this much (x w_guide); 0 = ours alone
    'guide_init': 'warm',  # with a guide: 'warm' = our frame-by-frame start with the guide's pull on (the body turned
                           # by our tracks), 'copy' = start from the guide's poses themselves
    'seen_from_outlines': False,  # a limb's SAM 2 outline counts as seeing it (smoothing, mirror, gait); on the bench's
                             # plain gallop it made the far legs worse (SAM 2 outlines jump legs there), so off
    'outline_seen': .3,    # a limb is seen in a frame when its SAM 2 outline shows this share of its typical size
    'outline_seen_frames': .5,  # a leg seen so in this share of the frames is never bent like its twin (mirror, gait)
    'w_floor': 3000.0,       # px-equivalent per metre under the floor
    'w_bend': 400.0,
    'w_depth': 200.0,        # px-equivalent per metre along the view axis
    'w_prop_inside': 3000.0,  # px-equivalent per metre of body inside a prop
    'w_prop_touch': 1000.0,  # px-equivalent per metre of gap between a lifted prop and the body
    'prop_lift': .03,        # a prop is lifted when its lowest point is this share of the creature's height off the floor
    'prop_gap': .01,         # allowed gap (share of the height) before the touch term pulls
    'smooth_body': 300.0,    # px-equivalent per rad / frame^2 (per metre / frame^2 x 10 for the offset)
    'smooth_legs': 60.0,
    'smooth_tail': 60.0,
    'hidden_below': .4,      # a part with fewer than this share of its points visible counts as (partly) hidden
    'hidden_smooth': 4.0,    # extra smoothing of a fully hidden part (x(1 + this))
    'rounds': 4,
    'lm_iters': 12,
    # orientation rescue (after the warm start): frames whose outline overlap falls under this share of the clip's
    # first frames' try the body turned over
    'rescue': True,
    'rescue_iou_share': .85,
    'rescue_keep': 5,        # turned-over candidates fitted per key frame (the best by their raw overlap)
    'rescue_step': 4,        # key frames: every n-th frame of a run that lost the outline
    'rescue_turn_cost': .25,  # per radian the body turns between key frames, against the outline overlap (0..1)
    'rescue_margin': .03,    # a turned body must win this much outline overlap on some probed frame to search the run
    'rescue_min_run': 10,    # a shorter run that ends back on the outline is a slip, not a body turned over
}


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def weights(cfg):
    w = dict(DEFAULTS)
    w.update(cfg.get('solve', {}))
    return w


def sample(field, pts):
    mx = (pts[:, 0] - .5).astype(np.float32).reshape(-1, 1)
    my = (pts[:, 1] - .5).astype(np.float32).reshape(-1, 1)
    return cv2.remap(field, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(len(pts))


class Problem:
    def __init__(self, creature, clip, cfg, tracks):
        self.cr, self.clip, self.w = creature, clip, weights(cfg)
        explicit = set(cfg.get('solve', {}))
        rig, mesh = creature.rig, creature.mesh
        self.rig = rig
        self.times = np.asarray(creature.times(clip), float)
        self.T = len(self.times)
        self.views = creature.views(clip)
        self.lo, self.hi, self.base = creature.bounds(clip)
        self.act = creature.active(clip)
        self.na = len(self.act)
        self.p0 = creature.p0(clip)
        self.fixed = self.p0.copy()
        self.prior_w = creature.prior(clip)[self.act]
        self.depth_axis = creature.depth_axis(clip)
        # Floor and depth weights are per metre of a ~1.8 m creature: scale them to the rig's own units.
        self.unit = float(getattr(creature, 'height', 1.8)) / 1.8
        tris = mesh['triangles']
        # ----- tracks
        self.tv = tris[tracks['tri']]                      # (K, 3) vertex ids
        self.bary = tracks['bary'].astype(float)
        self.u = tracks['tracks'].astype(float)            # (K, T, 2)
        self.vis = tracks['visible_p'].astype(float)       # (K, T)
        self.tview = tracks['view'].astype(int)
        self.track_bone = np.asarray(tracks['bone'])
        self.gait_w = None                                # (T, na) per-frame pull to gait_target, once found
        names = rig.names
        part_of_bone = {}
        for pname, ids in creature.parts().items():
            for b in ids:
                part_of_bone[int(b)] = pname
        self.part_names = ['body'] + list(creature.parts())
        self.tpart = np.array([self.part_names.index(part_of_bone.get(int(b), 'body')) for b in tracks['bone']])
        self.rel = np.ones(len(self.u))
        # A point on skin shared by several bones (blended weights) follows the rig only roughly: less weight.
        top = creature.wts.max(axis=1)[self.tv]                      # (K, 3)
        rigid = (self.bary * top).sum(1)
        self.rigid = np.clip((rigid - .5) / .4, .15, 1.0)
        # ----- silhouette samples: every 3rd vertex of the deforming mesh, part-aware weight
        nb = mesh.get('body_vertices', len(mesh['coords']))
        self.sil = np.arange(0, nb, int(self.w['sil_stride']))
        # ----- floor: the low vertices of the start pose (height along the creature's up axis)
        self.up = np.asarray(getattr(creature, 'up', (0.0, 0.0, 1.0)), float)
        self.floor_level = float(getattr(creature, 'floor', 0.0))
        X0 = rig.skin(self.p0, mesh['coords'][:nb], creature.idx[:nb], creature.wts[:nb])
        h0 = X0 @ self.up - self.floor_level
        span = max(float(h0.max() - h0.min()), 1e-6)
        body = creature.idx[:nb, 0] < getattr(rig, 'nb', 10 ** 9)
        self.floor = np.where((h0 < .05 * span) & body)[0][::2]
        verts = np.arange(len(mesh['coords']))            # all: the outline term may pick any triangle
        self.verts = verts
        look = -np.ones(len(mesh['coords']), int)
        look[verts] = np.arange(len(verts))
        self.l_tv, self.l_sil, self.l_floor = look[self.tv], look[self.sil], look[self.floor]
        # ----- props: their convex hull (rest pose), some of their vertices, the body samples that may touch them
        from scipy.spatial import ConvexHull
        self.height = float(getattr(creature, 'height', 1.8))
        self.props = []
        for k in range(len(getattr(rig, 'props', []))):
            pv = np.where(creature.idx[:, 0] == rig.nb + k)[0]
            if len(pv) < 8:
                continue
            hull = ConvexHull(mesh['coords'][pv])
            pv = pv[::max(1, len(pv) // 300)]
            self.props.append(dict(k=k, eq=hull.equations, pv=pv, l_pv=look[pv]))
        self.cbody = self.sil[creature.idx[self.sil, 0] < rig.nb] if self.props else np.zeros(0, int)
        self.l_cbody = look[self.cbody]
        self.act_col = {int(c): i for i, c in enumerate(self.act)}
        self.rest = mesh['coords'][verts]
        self.idx, self.wts = creature.idx[verts], creature.wts[verts]
        self.bends = creature.bends()
        # ----- silhouettes: distance outside (capped), per view and frame
        self.dout, self.vmask = [], []
        k = np.ones((2 * self.w['inside_dilate_px'] + 1,) * 2, np.uint8)
        for v in self.views:
            per, masks = [], []
            for i in range(self.T):
                raw = v['mask'](i)
                masks.append(raw)
                m = cv2.dilate(raw.astype(np.uint8), k)
                d = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5)
                per.append(np.minimum(d, self.w['inside_cap_px']).astype(np.float32))
            self.dout.append(per)
            self.vmask.append(masks)
        self.cover = [[None] * len(self.views) for _ in range(self.T)]
        self.tris = tris
        # each limb's own outline per frame (SAM 2), when installed: the tracks that sit on another limb set aside, and
        # points along every lower-limb bone (vertices of their own, rigid on the bone) for the centre line term
        self.plab = None
        self.colour_parts = False
        extra_pts = []                                    # (bone, place along it, 'axis' | 'flat', limb)
        if self.w['w_parts'] > 0:
            from .parts import segment, triangle_objects
            labels = None
            if self.w['colour_parts']:
                from .colours import segment as colour_segment
                labels, of, names, self.colour_info = colour_segment(creature, clip)
                if labels is not None:
                    # every limb is measured by its colour: none is filled in from its twin (mirror, gait), none is
                    # reached for by the far end of its outline (tip: the paint ends before a pale paw does)
                    self.colour_parts = True
                    for key, val in (('seen_from_outlines', True), ('mirror_legs', False), ('w_gait', 0.0),
                                     ('w_tip', 0.0), ('w_keypoints', 0.0)):
                        if key not in explicit:
                            self.w[key] = val
            if labels is None:
                labels, of, names = segment(creature, clip)
            if labels is not None:
                self.plab, self.tri_obj = labels, triangle_objects(creature, of)
                self.bone_obj, self.limb_seen = of, self.limb_visibility(labels)
                bones = np.asarray(tracks['bone'])
                self.track_obj = np.where(bones < rig.nb, of[np.minimum(bones, rig.nb - 1)], 1)
                self.parts_dropped = self.drop_tracks_off_their_limb()
                if self.w['w_axis'] > 0:
                    # bones that run along the leg (a paw or toes lying across it would be folded onto the line)
                    Hd, Tl = rig.bone_points(self.p0)
                    for j in range(rig.nb):
                        d = Tl[j] - Hd[j]
                        if j in creature.legs:                # a leg: the bones nearer vertical than 35 degrees
                            use = abs(d @ self.up) > np.cos(np.radians(35)) * np.linalg.norm(d)
                        else:                                 # an arm: all but the last (a hand lies across)
                            use = any(rig.parents[c] == j for c in range(rig.nb))
                        if of[j] >= 2 and rig.length[j] > 0 and use:
                            for f in (.2, .5, .8):
                                extra_pts.append((j, f, 'axis', of[j]))
                if self.w['w_tip'] > 0:
                    depth = lambda j: 0 if rig.parents[j] < 0 else 1 + depth(rig.parents[j])
                    for o in np.unique(of[of >= 2]):
                        mine = [j for j in range(rig.nb) if of[j] == o and rig.length[j] > 0
                                and not FINGER.search(rig.names[j])]
                        if mine:
                            extra_pts.append((max(mine, key=depth), 1.0, 'tip', o))
                self.part_area = [{o: float(np.median((lab_ == o).sum((1, 2)))) for o in np.unique(lab_[lab_ >= 2])}
                                  for lab_ in labels]
                # a four-legged animal: its elbows, knees and paws from an animal pose model
                if self.w['w_keypoints'] > 0:
                    from . import keypoints as kpmod
                    legs = kpmod.leg_map(creature)
                    found_kp = kpmod.detect(creature, clip) if legs else None
                    if found_kp is not None:
                        for key, chain in legs.items():
                            hinge = next((i for i, j in enumerate(chain) if rig.names[j] in creature.hinge),
                                         len(chain) // 2)
                            joint_i, paw_i = kpmod.LEGS[key]
                            extra_pts += [(chain[hinge], 0.0, 'kp', joint_i), (chain[-1], 0.0, 'kp', paw_i)]
                        self.kp_frames, self.kp_swaps = [], 0
                        for vi in range(len(labels)):
                            fr_, sw_ = kpmod.assigned(found_kp[vi], labels[vi], legs, of)
                            self.kp_frames.append(fr_)
                            self.kp_swaps += sw_
        # a foot on the floor lies as flat as at the start: the drop of each foot bone (the last two of a standing leg)
        if self.w['w_flat'] > 0:
            for chain in creature.limb_chains(as_list=True):
                if set(chain) & creature.legs:
                    for j in chain[-2:]:
                        if rig.length[j] > 0:
                            extra_pts += [(j, .1, 'flat', 0), (j, .9, 'flat', 0)]
        self.axis_v = np.zeros(0, int)
        self.tip_v = np.zeros(0, int)
        self.kp_v = np.zeros(0, int)
        self.flat = np.zeros((0, 2), int)
        if extra_pts:
            n0, K = len(self.rest), self.idx.shape[1]
            pts = np.array([rig.rest_head[j] + f * rig.length[j] * rig.rest_rot[j][:, 1] for j, f, _, _ in extra_pts])
            idx = np.zeros((len(pts), K), self.idx.dtype)
            idx[:, 0] = [j for j, _, _, _ in extra_pts]
            wts = np.zeros((len(pts), K), self.wts.dtype)
            wts[:, 0] = 1.0
            self.rest = np.vstack([self.rest, pts])
            self.idx, self.wts = np.vstack([self.idx, idx]), np.vstack([self.wts, wts])
            kind = np.array([k for _, _, k, _ in extra_pts])
            ids = n0 + np.arange(len(pts))
            self.axis_v = ids[kind == 'axis']
            self.axis_obj = np.array([o for _, _, k, o in extra_pts if k == 'axis'], int)
            self.axis_tri = len(self.tris) + np.arange(len(self.axis_v))
            self.tris = np.vstack([self.tris, np.repeat(self.axis_v[:, None], 3, 1)])
            self.tip_v = ids[kind == 'tip']
            self.tip_obj = np.array([o for _, _, k, o in extra_pts if k == 'tip'], int)
            self.tip_tri = len(self.tris) + np.arange(len(self.tip_v))
            self.tris = np.vstack([self.tris, np.repeat(self.tip_v[:, None], 3, 1)])
            self.kp_v = ids[kind == 'kp']
            kp_idx = [o for _, _, k, o in extra_pts if k == 'kp']
            self.kp_tri = {int(i): len(self.tris) + n for n, i in enumerate(kp_idx)}
            self.tris = np.vstack([self.tris, np.repeat(self.kp_v[:, None], 3, 1)])
            fl = ids[kind == 'flat'].reshape(-1, 2)
            X0e = rig.skin(self.p0, self.rest[fl.ravel()], self.idx[fl.ravel()], self.wts[fl.ravel()]).reshape(-1, 2, 3)
            self.flat = fl
            self.flat_drop = (X0e[:, 1] - X0e[:, 0]) @ self.up
            self.verts = np.arange(len(self.rest))
        # Jacobians only for the vertices a frame uses (tracks, silhouette samples, floor, its outline matches)
        self.need_base = np.unique(np.concatenate([self.tv.ravel(), self.sil, self.floor, self.axis_v, self.tip_v,
                                                   self.kp_v, self.flat.ravel()]
                                                  + [pr['pv'] for pr in self.props]))
        self.need = [self.need_base] * self.T
        self.tv_unique = np.unique(self.tv.ravel())
        # ----- smoothness: per channel weight, content-time second differences, firmer where the part is hidden
        self.smooth_w = np.array([self.channel_smooth(c) for c in self.act])
        self.smooth_mult = self.hidden_multiplier(part_of_bone)
        self.D = self.second_difference()
        # the graphics card computes every frame's terms at once when PyTorch has CUDA (props: the processor's path)
        self.gpu = None
        if device() is not None:
            from .gpu_terms import GpuTerms
            if GpuTerms.usable(self):
                self.gpu = GpuTerms(self, device())

    def normal(self, t, x_t, robust):
        """J'J, J'r and the cost of frame t."""
        if self.gpu is not None:
            H, g, c = self.gpu.frames([t], x_t[None], None if robust is None else robust[None])
            return H[0], g[0], float(c[0])
        r, J = self.frame_terms(t, x_t, robust)
        return J.T @ J, J.T @ r, float(r @ r)

    def frame_cost(self, t, x_t, robust):
        if self.gpu is not None:
            return float(self.gpu.frames([t], x_t[None], None if robust is None else robust[None], want_jac=False)[0])
        r = self.frame_terms(t, x_t, robust, want_jac=False)
        return float(r @ r)

    # ----- helpers --------------------------------------------------------------------------------------------
    def channel_smooth(self, c):
        if c < 3:
            return self.w['smooth_body'] * 10.0 / self.unit   # metres: 1 cm / frame^2 ~ 30 px-equivalent
        if c - 3 >= len(self.rig.channels):
            return self.w['smooth_legs']                  # extra channels (fists, wing folds): fast
        j, _ = self.rig.channels[c - 3]
        name = self.rig.names[j]
        limbs = self.cr.limbs() if hasattr(self.cr, 'limbs') else [b for v in self.cr.parts().values() for b in v]
        if name in limbs:
            return self.w['smooth_legs']
        if name.startswith('Tail'):
            return self.w['smooth_tail']
        return self.w['smooth_body']

    def limb_visibility(self, labels):
        """(objects, views, T): how much of each limb's outline the video shows in a frame, against the limb's own
        typical size in that view (90th percentile over the clip), from the SAM 2 labels. This, not the count of
        tracked points, says whether a limb is seen: points exist only where the limb showed on frame 0, so a leg
        hidden at the start had almost none for the whole clip while its outline was plain to see."""
        n = int(max(int(lab.max()) for lab in labels)) + 1
        out = np.zeros((max(n, 2), len(labels), self.T))
        for vi, lab in enumerate(labels):
            for o in range(2, n):
                a = (lab == o).sum((1, 2)).astype(float)
                out[o, vi] = np.clip(a / max(np.percentile(a, 90), 100.0), 0, 1)
        return out

    def side_on(self, lateral):
        """Per view: how side-on it looks at the creature (1 sees a leg swing forward and back, 0 looks along it)."""
        return np.array([abs(v['cam'].fwd @ v['cam'].Ma @ lateral) ** 2 for v in self.views])

    def outline_seen(self, chain, side=None):
        """(T,): how well the outlines show the lower part of a limb chain per frame (best view, each view weighted by
        side), or None without SAM 2 labels."""
        if self.plab is None or not self.w['seen_from_outlines']:
            return None
        o = int(self.bone_obj[chain[-1]])
        if o < 2 or o >= len(self.limb_seen):
            return None
        s = self.limb_seen[o] * (np.ones(len(self.views)) if side is None else side)[:, None]
        return s.max(0)

    def hidden_multiplier(self, part_of_bone):
        """(T, na): 1 where a channel's part shows enough of its tracked points or of its own outline, up to
        1 + hidden_smooth where it shows neither (its share of visible points against its best frame, or of its
        outline against its typical size)."""
        w = self.w
        seen = self.vis >= w['visible_min']
        share = np.ones((len(self.part_names), self.T))
        parts = self.cr.parts()
        for pi in range(len(self.part_names)):
            mine = self.tpart == pi
            if mine.any():
                cnt = seen[mine].sum(0).astype(float)
                share[pi] = cnt / max(cnt.max(), 1.0)
            bones = [b for b in parts.get(self.part_names[pi], []) if b < self.rig.nb]
            ol = self.outline_seen(bones) if bones else None
            if ol is not None:
                share[pi] = np.maximum(share[pi], ol)
        extra = w['hidden_smooth'] * np.clip(1 - share / max(w['hidden_below'], 1e-6), 0, 1)
        rig = self.rig
        out = np.ones((self.T, self.na))
        for i, c in enumerate(self.act):
            if c < 3:
                part = 'body'
            elif c < getattr(rig, 'prop0', 10 ** 9):
                part = part_of_bone.get(int(rig.channels[c - 3][0]), 'body')
            else:
                k = (c - rig.prop0) // 6
                part = 'prop:' + rig.props[k]['name']
            if part in self.part_names and part != 'body':
                out[:, i] += extra[self.part_names.index(part)]
        return out

    def second_difference(self):
        """Rows (T-2) x na of accelerations in units of the clip's median frame step."""
        T, na = self.T, self.na
        t = self.times
        dt0 = float(np.median(np.diff(t)))
        rows, cols, vals = [], [], []
        r = 0
        for i in range(1, T - 1):
            dm, dp = t[i] - t[i - 1], t[i + 1] - t[i]
            cm = dt0 * dt0 * 2 / (dm * (dm + dp))
            cp = dt0 * dt0 * 2 / (dp * (dm + dp))
            c0 = -(cm + cp)
            for c in range(na):
                for ii, cc in ((i - 1, cm), (i, c0), (i + 1, cp)):
                    rows.append(r); cols.append(ii * na + c); vals.append(cc * self.smooth_w[c] * self.smooth_mult[i, c])
                r += 1
        return sp.csr_matrix((vals, (rows, cols)), shape=(r, T * na))

    def full(self, x_t):
        p = self.fixed.copy()
        p[self.act] = x_t
        return p

    # ----- one frame -------------------------------------------------------------------------------------------
    def frame_terms(self, t, x_t, robust=None, want_jac=True):
        """Residual vector and Jacobian (active columns) of frame t; robust = per-track IRLS weights."""
        rig = self.rig
        p = self.full(x_t)
        # Skin only the vertices this frame uses (tracks, silhouette samples, floor, outline matches): a 50k-vertex
        # character skinned whole at every step made the solve 5x slower.
        need = self.need[t]
        X = np.zeros((len(self.rest), 3))
        J = pos = None
        if want_jac:
            Xn, J = rig.skin(p, self.rest[need], self.idx[need], self.wts[need], jac=True)
            J = J[:, :, self.act]
            pos = np.full(len(self.rest), -1)
            pos[need] = np.arange(len(need))
        else:
            Xn = rig.skin(p, self.rest[need], self.idx[need], self.wts[need])
        X[need] = Xn
        res, jac = [], []
        w = self.w
        # tracks
        for vi, v in enumerate(self.views):
            sel = (self.tview == vi) & (self.vis[:, t] >= w['visible_min'])
            ids = np.where(sel)[0]
            if len(ids):
                bw = self.bary[ids]
                pts = np.einsum('km,kmd->kd', bw, X[self.l_tv[ids]])
                pix, dP = project(v, pts, jac=True)
                e = pix - self.u[ids, t]
                ww = self.vis[ids, t] * self.rel[ids] * self.rigid[ids]
                if robust is not None:
                    ww = ww * robust[ids]
                for pi in range(len(self.part_names)):
                    m = self.tpart[ids] == pi
                    if not m.any():
                        continue
                    norm = max(m.sum(), w['track_min_points'])
                    s = np.sqrt(ww[m] * w['w_track'] / norm)
                    res.append((e[m] * s[:, None]).ravel())
                    if want_jac:
                        Jp = np.einsum('km,kmdn->kdn', bw[m], J[pos[self.l_tv[ids[m]]]])
                        jac.append((np.einsum('kad,kdn->kan', dP[m], Jp) * s[:, None, None]).reshape(-1, self.na))
            # inside the silhouette
            pts = X[self.l_sil]
            pix, dP = project(v, pts, jac=True)
            d = sample(self.dout[vi][t], pix)
            s = np.sqrt(w['w_inside'] / len(pix))
            res.append(d * s)
            if want_jac:
                gx = (sample(self.dout[vi][t], pix + [1, 0]) - sample(self.dout[vi][t], pix - [1, 0])) / 2
                gy = (sample(self.dout[vi][t], pix + [0, 1]) - sample(self.dout[vi][t], pix - [0, 1])) / 2
                g = np.einsum('ka,kad->kd', np.stack([gx, gy], 1), dP)   # d distance / d point (3)
                jac.append(np.einsum('kd,kdn->kn', g, J[pos[self.l_sil]]) * s)
            # video silhouette the model leaves uncovered: pull the nearest model outline point there
            cv_ = self.cover[t][vi]
            if cv_ is not None and len(cv_['target']):
                tv = self.tris[cv_['tri']]
                pts = np.einsum('km,kmd->kd', cv_['bary'], X[tv])
                pix, dPc = project(v, pts, jac=True)
                e = pix - cv_['target']
                c = cv_['c'] if 'c' in cv_ else w['cover_cauchy_px']
                rw = 1 / (1 + (np.linalg.norm(e, axis=1) / c) ** 2)
                s = np.sqrt(rw * (cv_['w'] if 'w' in cv_ else w['w_cover'] / max(len(e), w['cover_min_points'])))
                res.append((e * s[:, None]).ravel())
                if want_jac:
                    Jp = np.einsum('km,kmdn->kdn', cv_['bary'], J[pos[tv]])
                    jac.append((np.einsum('kad,kdn->kan', dPc, Jp) * s[:, None, None]).reshape(-1, self.na))
        # a foot on the floor lies as flat as at the start
        if len(self.flat):
            za, zb = X[self.flat[:, 0]] @ self.up, X[self.flat[:, 1]] @ self.up
            on = np.minimum(za, zb) - self.floor_level < w['contact'] * self.height
            k_ = w['w_flat'] / self.unit
            res.append(((zb - za) - self.flat_drop) * on * k_)
            if want_jac:
                jac.append(np.einsum('d,kdn->kn', self.up, J[pos[self.flat[:, 1]]] - J[pos[self.flat[:, 0]]])
                           * (on * k_)[:, None])
        # floor
        z = X[self.l_floor] @ self.up - self.floor_level
        below = z < 0
        res.append(np.where(below, z, 0.0) * w['w_floor'] / self.unit)
        if want_jac:
            jac.append(np.einsum('d,kdn->kn', self.up, J[pos[self.l_floor]]) * below[:, None] * w['w_floor'] / self.unit)
        # props: no body inside them; resting on the body once lifted
        for pr in self.props:
            rr, jj = self.prop_terms(pr, p, X, J, pos, want_jac)
            res.append(rr)
            if want_jac:
                jac.append(jj)
        # prior + depth
        dp = (p - self.base)[self.act]
        res.append(dp * self.prior_w)
        if want_jac:
            jac.append(np.diag(self.prior_w))
        if self.gait_w is not None:
            res.append((x_t - self.gait_target[t]) * self.gait_w[t])
            if want_jac:
                jac.append(np.diag(self.gait_w[t]))
        depth_row = np.zeros(self.na)
        tr_cols = [list(self.act).index(c) for c in (0, 1, 2) if c in self.act]
        if len(tr_cols) == 3 and self.depth_axis is not None:      # one camera: depth is not seen
            depth_row[tr_cols] = self.depth_axis
        res.append(np.array([depth_row @ (x_t - self.p0[self.act]) * w['w_depth'] / self.unit]))
        if want_jac:
            jac.append((depth_row * w['w_depth'] / self.unit)[None])
        # bend directions
        for (a_j, b_j, c_j, sign, margin) in self.bends:
            pts = np.stack([rig.heads[a_j], rig.heads[b_j], rig.heads[c_j]])
            ids3 = np.array([[a_j] * 4, [b_j] * 4, [c_j] * 4])
            w1 = np.zeros((3, 4)); w1[:, 0] = 1
            if want_jac:
                bp, bj = rig.skin(p, pts, ids3, w1, jac=True)
                bj = bj[:, :, self.act]
            else:
                bp = rig.skin(p, pts, ids3, w1)
            u, v = bp[1] - bp[0], bp[2] - bp[1]
            Y = np.array([0., 1., 0.])
            scale = sign / (np.linalg.norm(u) * np.linalg.norm(v))
            val = np.cross(u, v) @ Y * scale - margin
            on = val < 0
            res.append(np.array([val * w['w_bend'] if on else 0.0]))
            if want_jac:
                grad = (np.cross(v, Y) @ (bj[1] - bj[0]) + np.cross(Y, u) @ (bj[2] - bj[1])) * scale
                jac.append((grad * (w['w_bend'] if on else 0.0))[None])
        r = np.concatenate(res)
        return (r, np.vstack(jac)) if want_jac else r

    def prop_terms(self, pr, p, X, J, pos, want_jac):
        """Signed distance of the body samples to the prop's convex hull (rest frame: max over its planes, negative
        inside). Every sample inside is pushed out; when the prop is off the floor, the nearest sample is pulled onto
        its surface. Jacobian: the body through the samples' own, the prop through its translation and turn."""
        w, rig = self.w, self.rig
        k = pr['k']
        base = rig.prop0 + 6 * k
        R = rig.prop_rotation(p, k)
        c0, tt = rig.prop_centre[k], p[base:base + 3]
        xb = X[self.l_cbody]
        rel = xb - c0 - tt
        sp_ = (rel @ R + c0) @ pr['eq'][:, :3].T + pr['eq'][:, 3]
        j = sp_.argmax(1)
        sd = sp_[np.arange(len(xb)), j]
        m = pr['eq'][j, :3] @ R.T                                   # the active plane's normal in the world
        lowest = (X[pr['l_pv']] @ self.up).min() - self.floor_level
        lifted = lowest > w['prop_lift'] * self.height
        inside = sd < 0
        rows = np.where(inside)[0]
        gap = 0.0
        touch = None
        if lifted and not inside.any():
            touch = int(sd.argmin())
            gap = max(sd[touch] - w['prop_gap'] * self.height, 0.0)
        s_in, s_t = w['w_prop_inside'] / self.unit, w['w_prop_touch'] / self.unit
        r = np.concatenate([sd[rows] * s_in / np.sqrt(max(len(rows), 1)), [gap * s_t]])
        if not want_jac:
            return r, None
        a, b = p[base + 3], p[base + 4]
        ax = [np.array([1.0, 0, 0]), rot_x(a) @ np.array([0, 1.0, 0]), rot_x(a) @ rot_y(b) @ np.array([0, 0, 1.0])]

        def grad(i):
            g = m[i] @ J[pos[self.l_cbody[i]]]                       # through the body
            for c, v in ((base, -m[i, 0]), (base + 1, -m[i, 1]), (base + 2, -m[i, 2])):
                if c in self.act_col:
                    g[self.act_col[c]] += v
            for q in range(3):
                c = base + 3 + q
                if c in self.act_col:
                    g[self.act_col[c]] += ax[q] @ np.cross(m[i], rel[i])
            return g
        Jr = np.zeros((len(r), self.na))
        for n_, i in enumerate(rows):
            Jr[n_] = grad(i) * s_in / np.sqrt(max(len(rows), 1))
        if touch is not None and gap > 0:
            Jr[-1] = grad(touch) * s_t
        return r, Jr

    def track_errors(self, x):
        """Pixel error of every track in every frame (nan where unseen)."""
        E = np.full(self.vis.shape, np.nan)
        for t in range(self.T):
            p = self.full(x[t])
            X = np.zeros((len(self.rest), 3))
            tvu = self.tv_unique
            X[tvu] = self.rig.skin(p, self.rest[tvu], self.idx[tvu], self.wts[tvu])
            for vi, v in enumerate(self.views):
                ids = np.where((self.tview == vi) & (self.vis[:, t] >= self.w['visible_min']))[0]
                pts = np.einsum('km,kmd->kd', self.bary[ids], X[self.l_tv[ids]])
                E[ids, t] = np.linalg.norm(project(v, pts) - self.u[ids, t], axis=1)
        return E

    def update_cover(self, x, frames=None):
        """Per frame and view: video silhouette pixels the posed model misses (beyond 2 px), each matched to the
        nearest pixel of the model's outline (its triangle and barycentric point), like ICP. frames: only these (x
        indexed by frame)."""
        from scipy.spatial import cKDTree
        k3 = np.ones((5, 5), np.uint8)
        total = 0
        for t in (range(self.T) if frames is None else frames):
            p = self.full(x[t])
            X = self.pose_all(p)
            for vi, v in enumerate(self.views):
                pix = project(v, X)
                tri, bary, _ = self.cr.rasterize(pix, view_depth(v, X), self.tris, v['size'])
                model = tri >= 0
                miss = self.vmask[vi][t] & ~cv2.dilate(model.astype(np.uint8), k3).astype(bool)
                miss = cv2.morphologyEx(miss.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
                ys, xs = np.nonzero(miss)
                found = []
                if len(ys) >= self.w['cover_min_area']:
                    keep = (ys % 2 == 0) & (xs % 2 == 0)
                    ys, xs = ys[keep], xs[keep]
                    inner = cv2.erode(model.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    by, bx = np.nonzero(model & ~inner)
                    d, nn = cKDTree(np.stack([bx, by], 1)).query(np.stack([xs, ys], 1))
                    ok = d < self.w['cover_reach_px']
                    n = int(ok.sum())
                    found.append((tri[by[nn[ok]], bx[nn[ok]]], bary[by[nn[ok]], bx[nn[ok]]],
                                  np.stack([xs[ok] + .5, ys[ok] + .5], 1).astype(float),
                                  np.full(n, self.w['w_cover'] / max(n, self.w['cover_min_points'])),
                                  np.full(n, self.w['cover_cauchy_px'])))
                if self.plab is not None:
                    found += self.part_matches(vi, t, tri, bary, pix)
                if len(self.kp_v):
                    kps = [k_ for k_ in self.kp_frames[vi][t] if k_[0] in self.kp_tri]
                    if kps:
                        b1 = np.zeros((len(kps), 3))
                        b1[:, 0] = 1.0
                        found.append((np.array([self.kp_tri[k_[0]] for k_ in kps]), b1,
                                      np.array([[k_[1], k_[2]] for k_ in kps]),
                                      np.array([self.w['w_keypoints'] * k_[3] for k_ in kps]),
                                      np.full(len(kps), self.w['keypoint_cauchy_px'])))
                if not found or not sum(len(f[0]) for f in found):
                    self.cover[t][vi] = None
                    continue
                self.cover[t][vi] = dict(tri=np.concatenate([f[0] for f in found]),
                                         bary=np.concatenate([f[1] for f in found]),
                                         target=np.concatenate([f[2] for f in found]),
                                         w=np.concatenate([f[3] for f in found]),
                                         c=np.concatenate([f[4] for f in found]))
                total += len(self.cover[t][vi]['tri'])
            extra = [self.tris[c['tri']].ravel() for c in self.cover[t] if c is not None]
            self.need[t] = np.unique(np.concatenate([self.need_base] + extra)) if extra else self.need_base
        return total

    def drop_tracks_off_their_limb(self, near_px=3):
        """A limb's track that stands on another limb's outline, or off the creature, in a frame is not counted
        there: the tracker slid from one leg onto the next (four legs of the same fur crossing)."""
        k = np.ones((2 * near_px + 1,) * 2, np.uint8)
        dropped = 0
        for vi, labs in enumerate(self.plab):
            ids = np.where((self.tview == vi) & (self.track_obj >= 2))[0]
            if not len(ids):
                continue
            H, W = labs.shape[1:]
            for t in range(self.T):
                lab = labs[t]
                c = np.clip(np.floor(self.u[ids, t, 0]).astype(int), 0, W - 1)
                r = np.clip(np.floor(self.u[ids, t, 1]).astype(int), 0, H - 1)
                at = lab[r, c]
                bad = (at != self.track_obj[ids]) & (at != 1)        # on another limb, or off the creature
                for obj in np.unique(self.track_obj[ids][bad]):     # a few pixels off its own outline is fine
                    sel = bad & (self.track_obj[ids] == obj)
                    near = cv2.dilate((lab == obj).astype(np.uint8), k).astype(bool)
                    bad[sel] = ~near[r[sel], c[sel]]
                hit = ids[bad & (self.vis[ids, t] >= self.w['visible_min'])]
                self.vis[hit, t] = 0.0
                dropped += len(hit)
        return dropped

    def part_matches(self, vi, t, tri, bary, pix=None):
        """Outline matches of every limb in one view and frame: its outline pixels the posed limb misses, each to the
        nearest pixel of the limb's own edge, and the limb's visible pixels outside its outline, each to the nearest
        pixel of that outline. (tri, bary, target, weight) lists."""
        from scipy.spatial import cKDTree
        lab = self.plab[vi][t]
        obj = np.where(tri >= 0, self.tri_obj[np.maximum(tri, 0)], 0)
        k5, k3 = np.ones((5, 5), np.uint8), np.ones((3, 3), np.uint8)
        out = []
        for o in np.unique(lab[lab >= 2]):
            mk, mm = lab == o, obj == o
            if mk.sum() < 30 or mm.sum() < 30:
                continue                                          # hidden in the video or in the pose
            edge_m = mm & ~cv2.erode(mm.astype(np.uint8), k3).astype(bool)
            edge_k = mk & ~cv2.erode(mk.astype(np.uint8), k3).astype(bool)
            tri_l, bary_l, tgt = [], [], []
            ys, xs = np.nonzero(mk & ~cv2.dilate(mm.astype(np.uint8), k5).astype(bool))
            sel = (ys % 2 == 0) & (xs % 2 == 0)
            ys, xs = ys[sel], xs[sel]
            if len(ys):
                by, bx = np.nonzero(edge_m)
                d, nn = cKDTree(np.stack([bx, by], 1)).query(np.stack([xs, ys], 1))
                ok = d < self.w['cover_reach_px']
                tri_l.append(tri[by[nn[ok]], bx[nn[ok]]]); bary_l.append(bary[by[nn[ok]], bx[nn[ok]]])
                tgt.append(np.stack([xs[ok] + .5, ys[ok] + .5], 1))
            ys, xs = np.nonzero(mm & ~cv2.dilate(mk.astype(np.uint8), k5).astype(bool))
            sel = (ys % 2 == 0) & (xs % 2 == 0)
            ys, xs = ys[sel], xs[sel]
            if len(ys):
                ky, kx = np.nonzero(edge_k)
                d, nn = cKDTree(np.stack([kx, ky], 1)).query(np.stack([xs, ys], 1))
                ok = d < self.w['cover_reach_px']
                tri_l.append(tri[ys[ok], xs[ok]]); bary_l.append(bary[ys[ok], xs[ok]])
                tgt.append(np.stack([kx[nn[ok]] + .5, ky[nn[ok]] + .5], 1))
            n = sum(len(a) for a in tri_l)
            if n:
                out.append((np.concatenate(tri_l), np.concatenate(bary_l), np.concatenate(tgt).astype(float),
                            np.full(n, self.w['w_parts'] / max(n, self.w['cover_min_points'])),
                            np.full(n, self.w['parts_cauchy_px'])))
            # the limb's bones onto the centre line of its outline
            mine = np.where(self.axis_obj == o)[0] if len(self.axis_v) and pix is not None else []
            if len(mine):
                from skimage.morphology import skeletonize
                sy, sx = np.nonzero(skeletonize(mk))
                if len(sy) >= 5:
                    q = pix[self.axis_v[mine]]
                    d, nn = cKDTree(np.stack([sx, sy], 1) + .5).query(q)
                    ok = d < self.w['cover_reach_px']
                    if ok.any():
                        bary1 = np.zeros((int(ok.sum()), 3))
                        bary1[:, 0] = 1.0
                        out.append((self.axis_tri[mine[ok]], bary1, np.stack([sx[nn[ok]], sy[nn[ok]]], 1) + .5,
                                    np.full(int(ok.sum()), self.w['w_axis'] / max(int(ok.sum()), 8)),
                                    np.full(int(ok.sum()), self.w['parts_cauchy_px'])))
            # the limb's end: the point of its outline farthest from the trunk's middle (a whole, unhidden limb only;
            # where the limb meets the body is no anchor: a galloping leg lies along the belly for half its length)
            tip = np.where(self.tip_obj == o)[0] if len(self.tip_v) else []
            if len(tip) and mk.sum() > .4 * self.part_area[vi].get(o, 0):
                by_, bx_ = np.nonzero(lab == 1)
                n_, cc_, st_, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), 8)
                if len(by_) >= 50 and n_ > 1:
                    ys, xs = np.nonzero(cc_ == 1 + int(np.argmax(st_[1:, cv2.CC_STAT_AREA])))   # its main piece
                    far = int(np.argmax((xs - bx_.mean()) ** 2 + (ys - by_.mean()) ** 2))
                    bary1 = np.zeros((1, 3))
                    bary1[0, 0] = 1.0
                    out.append((self.tip_tri[tip[:1]], bary1, np.array([[xs[far] + .5, ys[far] + .5]]),
                                np.array([self.w['w_tip']]), np.array([self.w['tip_cauchy_px']])))
        return out

    def mirror_legs(self, x, log):
        """Pairs of standing legs that mirror each other across the body's middle plane; where one has three times
        the other's tracked points, the other's bends are held within the range the first used over the clip (signs
        through the mirror). Returns True when a range changed."""
        cr, rig, w = self.cr, self.rig, self.w
        chains = [c for c in cr.limb_chains(as_list=True) if set(c) & cr.legs]
        lateral = np.cross(self.up, cr.facing)
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        mirror = lambda v: v - 2 * (v @ lateral) * lateral
        Hd, _ = rig.bone_points(self.p0)
        centre = Hd[rig.root]
        _, _, axes = rig.fk(self.p0)
        axis_of = {k: ax for _, ax, k in axes}
        # points seen by the cameras that see a leg swing forward and back (side-on); a front camera cannot tell
        side = self.side_on(lateral)
        seen = self.vis >= w['visible_min']
        count = [int(sum(side[vi] * seen[(self.tview == vi) & np.isin(self.track_bone, c)].sum()
                         for vi in range(len(self.views)))) for c in chains]
        col = {int(c): i for i, c in enumerate(self.act)}
        changed, used = False, set()
        for i, ci in enumerate(chains):
            if i in used:
                continue
            twin = min((k for k in range(len(chains)) if k != i and len(chains[k]) == len(ci) and k not in used),
                       key=lambda k: np.linalg.norm(centre + mirror(Hd[ci[0]] - centre) - Hd[chains[k][0]]), default=None)
            if twin is None or np.linalg.norm(centre + mirror(Hd[ci[0]] - centre) - Hd[chains[twin][0]]) > .25 * cr.height:
                continue
            used |= {i, twin}
            good, poor = (i, twin) if count[i] >= count[twin] else (twin, i)
            if count[good] < 3 * max(count[poor], 1):
                continue
            # few tracked points is not a hidden leg: the side cameras' outlines decide (a leg seen in most frames
            # follows its own outline, its twin's bends would overwrite what the video shows)
            ol = self.outline_seen(chains[poor], side)
            if ol is not None and (ol > w['outline_seen']).mean() >= w['outline_seen_frames']:
                log(f'mirror: {rig.names[chains[poor][0]]} leg has few tracked points ({count[poor]}) but its outline '
                    f'shows in {int((ol > w["outline_seen"]).sum())}/{self.T} frames: it follows its own outline')
                continue
            held = []
            for jg, jp in zip(chains[good], chains[poor]):
                if jp not in rig.chan_of or jg not in rig.chan_of:
                    continue
                kg = rig.index(rig.names[jg], cr.bend_axis(jg))
                kp = rig.index(rig.names[jp], cr.bend_axis(jp))
                if kg not in col or kp not in col:
                    continue
                s = -np.sign(axis_of[kp] @ mirror(axis_of[kg])) or 1.0
                b = s * (x[:, col[kg]] - self.p0[kg])
                m = np.radians(w['mirror_margin_deg'])
                lo, hi = self.p0[kp] + np.percentile(b, 2) - m, self.p0[kp] + np.percentile(b, 98) + m
                if lo > self.lo[kp] or hi < self.hi[kp]:
                    self.lo[kp], self.hi[kp] = max(self.lo[kp], lo), min(self.hi[kp], hi)
                    held.append(rig.names[jp])
                    changed = True
            if held:
                log(f'mirror: {rig.names[chains[poor][0]]} leg ({count[poor]} tracked points) bends like '
                    f'{rig.names[chains[good][0]]} ({count[good]}): {", ".join(held)}')
            if w['w_gait'] > 0:
                changed |= self.gait(x, chains[good], chains[poor], axis_of, mirror, lateral, col, log)
        return changed

    def gait(self, x, good, poor, axis_of, mirror, lateral, col, log):
        """If the well seen leg's swing repeats (a walk, a trot, a run), the poorly seen twin is pulled, in the frames
        the side-on cameras lose it, toward the twin's own angles at the same point of the cycle (its phase found in
        the frames it is seen; half a cycle when never)."""
        rig, w, T = self.rig, self.w, self.T
        pairs = []
        for jg, jp in zip(good, poor):
            if jg not in rig.chan_of or jp not in rig.chan_of:
                continue
            for (ag, kg), (ap, kp) in zip(sorted(rig.chan_of[jg]), sorted(rig.chan_of[jp])):
                if kg in col and kp in col:
                    dot = axis_of[kp] @ mirror(axis_of[kg])
                    if abs(dot) > .5:
                        pairs.append((col[kg], col[kp], -np.sign(dot)))
        if not pairs:
            return False
        # the swing: the first two bones' bend of the good leg, its period from the autocorrelation
        swing = sum(x[:, col[rig.index(rig.names[j], self.cr.bend_axis(j))]] for j in good[:2]
                    if rig.index(rig.names[j], self.cr.bend_axis(j)) in col)
        sig = swing - swing.mean()
        ac = np.array([sig[:-L] @ sig[L:] / max(np.sqrt((sig[:-L] @ sig[:-L]) * (sig[L:] @ sig[L:])), 1e-9)
                       for L in range(6, T // 2)])
        if not len(ac) or ac.max() < w['gait_min_corr']:
            return False
        L = 6 + int(ac.argmax())
        # how well the side-on cameras see the poor leg in each frame (tracks seen, weighted by how side-on)
        side = self.side_on(lateral)
        lower = poor[len(poor) // 2:]                     # the part that hides behind the other leg
        seen = (self.vis >= w['visible_min']) & np.isin(self.track_bone, lower)[:, None]
        q = sum(side[vi] * seen[self.tview == vi].sum(0) for vi in range(len(self.views)))
        trust = np.clip(q / max(np.percentile(q, 90), 1.0), 0, 1)
        ol = self.outline_seen(poor, side)               # where its outline shows, the leg follows the outline
        if ol is not None:
            trust = np.maximum(trust, np.clip(ol / max(w['outline_seen'], 1e-6), 0, 1))
        near = lambda t_: t_ if t_ < T else t_ - L * ((t_ - T) // L + 1)
        G = np.array([pg for pg, _, _ in pairs])
        Pp = np.array([pp for _, pp, _ in pairs])
        S = np.array([s for _, _, s in pairs])
        # the twin's place in the cycle from the video itself: where the two lower legs stand forward and back in
        # the most side-on view (SAM 2 outlines), over the frames both are clearly seen; half a cycle without them
        phase, how = L // 2, 'half a cycle'
        if self.plab is not None:
            from .parts import bone_objects
            of, _ = bone_objects(self.cr)
            vi = int(np.argmax(side))
            lab = self.plab[vi]
            fwd = self.views[vi]['cam'].A @ self.cr.facing if self.views[vi]['cam'].type == 'ortho' else None
            if fwd is not None and of[good[-1]] >= 2 and of[poor[-1]] >= 2:
                fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
                ys, xs = np.mgrid[0:lab.shape[1], 0:lab.shape[2]]
                def place(k):
                    a_ = np.array([(lab[t] == k).sum() for t in range(T)], float)
                    c_ = np.array([(xs[lab[t] == k] * fwd[0] + ys[lab[t] == k] * fwd[1]).mean() if a_[t] > 30
                                   else np.nan for t in range(T)])
                    return c_, a_ > .3 * np.median(a_)
                cg, okg = place(of[good[-1]])
                cp, okp = place(of[poor[-1]])
                scores = []
                for f in range(L):
                    idx = np.array([near(t + f) for t in range(T)])
                    ok = okp & okg[idx] & ~np.isnan(cp) & ~np.isnan(cg[idx])
                    if ok.sum() >= max(8, L // 2):
                        a_, b_ = cp[ok] - cp[ok].mean(), cg[idx][ok] - cg[idx][ok].mean()
                        scores.append((a_ @ b_ / max(np.sqrt((a_ @ a_) * (b_ @ b_)), 1e-9), f))
                if scores and max(scores)[0] > .5:
                    phase, how = max(scores)[1], f'from the outlines (match {max(scores)[0]:.2f})'
        idx = [near(t + phase) for t in range(T)]
        if self.gait_w is None:
            self.gait_w = np.zeros((T, self.na))
            self.gait_target = np.zeros((T, self.na))
        self.gait_target[:, Pp] = self.p0[self.act][Pp] + S * (x[idx][:, G] - self.p0[self.act][G])
        self.gait_w[:, Pp] = w['w_gait'] * (1 - trust)[:, None]
        log(f'gait: {rig.names[good[0]]} repeats every {L} frames (match {ac.max():.2f}); {rig.names[poor[0]]} '
            f'follows it {phase} frames on, {how}, where the side cameras lose it ({int((trust < .5).sum())} frames)')
        return True

    def robust_weights(self, E_t, scale=1.0):
        c = self.w['cauchy_px'] * scale
        return 1 / (1 + (np.nan_to_num(E_t, nan=0.0) / c) ** 2)

    # ----- orientation rescue ----------------------------------------------------------------------------------
    def pose_all(self, p):
        """Every vertex of the posed mesh (on the card when there is one)."""
        if self.gpu is not None:
            return self.gpu.pose_all(p[self.act])
        return self.rig.skin(p, self.rest, self.idx, self.wts)

    def outline_iou(self, t, p):
        """How well pose p covers frame t's silhouette: overlap (IoU), averaged over the views."""
        X = self.pose_all(p)
        out = []
        for vi, v in enumerate(self.views):
            tri, _, _ = self.cr.rasterize(project(v, X), view_depth(v, X), self.tris, v['size'])
            m, vm = tri >= 0, self.vmask[vi][t]
            out.append((m & vm).sum() / max((m | vm).sum(), 1))
        return float(np.mean(out))

    def trunk_iou(self, t, p):
        """How well the posed trunk (the body without its lower limbs) covers the trunk's own SAM 2 outline."""
        X = self.pose_all(p)
        out = []
        for vi, v in enumerate(self.views):
            tri, _, _ = self.cr.rasterize(project(v, X), view_depth(v, X), self.tris, v['size'])
            m = (tri >= 0) & (self.tri_obj[np.clip(tri, 0, len(self.tri_obj) - 1)] == 1)
            vm = self.plab[vi][t] == 1
            out.append((m & vm).sum() / max((m | vm).sum(), 1))
        return float(np.mean(out))

    def fit_outline(self, t, x, prev=None, rounds=2, iters=3):
        """Frame t fitted to its silhouette alone (inside it and covering it), the tracks set aside: once a body turns
        over, the tracker carries points of its back onto its belly and they hold the wrong side up."""
        lo, hi = self.lo[self.act], self.hi[self.act]
        keep = self.vis[:, t].copy()
        self.vis[:, t] = 0.0
        try:
            for _ in range(rounds):
                if self.w['w_cover'] > 0:
                    self.update_cover({t: x}, [t])
                x = self.lm_frame(t, x, prev, lo, hi, iters=iters)
        finally:
            self.vis[:, t] = keep
        return x

    def root_cols(self):
        rig = self.rig
        cols = [self.act_col.get(rig.index(rig.names[rig.root], a)) for a in range(3)] if rig.root in rig.chan_of else []
        return cols if cols and None not in cols else None

    @staticmethod
    def near_rotvec(w, ref):
        """The rotation vector of w's rotation that is nearest to ref (w and w + 2 pi k along its axis turn alike)."""
        th = float(np.linalg.norm(w))
        if th < 1e-9:
            u = ref / max(np.linalg.norm(ref), 1e-9)
            opts = [u * 2 * np.pi * k for k in (-1, 0, 1)]
        else:
            u = w / th
            opts = [u * (th + 2 * np.pi * k) for k in (-2, -1, 0, 1)]
        return min(opts, key=lambda o: float(np.linalg.norm(o - ref)))

    def turn_candidates(self, t, base):
        """Frame t's body turned over: half and quarter turns of its own body and of the start pose's, each with the
        frame's limbs and with the start pose's (the lost frames' limbs are tangled; a body turned over with its limbs
        as at the start is a dead bug on its back, a dog rolled on its side), each moved so the body's middle stays
        where the frame had it (the root bone may sit at one end of the body). The best few by overlap fitted to the
        outline: (params, overlap, body turn matrix, overlap before the fit), the frame's own pose first."""
        ri = self.root_cols()
        lo, hi = self.lo[self.act], self.hi[self.act]
        start = np.clip(self.p0[self.act], lo, hi)
        turns = [np.zeros(3)] + [np.pi * e for e in np.eye(3)] + [s * np.pi / 2 * e for e in np.eye(3) for s in (1, -1)]
        ang = lambda R1, R2: float(np.linalg.norm(Rotation.from_matrix(R1.T @ R2).as_rotvec()))
        sub = slice(None, None, max(1, len(self.rest) // 600))
        middle = lambda xx: self.rig.skin(self.full(xx), self.rest[sub], self.idx[sub], self.wts[sub]).mean(0)
        w0 = base[ri]
        c0 = middle(base)
        raw, seen = [], []
        for R_ref in (exp_so3(w0), exp_so3(start[ri])):
            for li, limbs in enumerate((base, start)):
                for d in turns:
                    R = R_ref @ exp_so3(d)
                    if any(lj == li and ang(Rj, R) < .15 for lj, Rj in seen):
                        continue                             # the same turn reached from the other side
                    seen.append((li, R))
                    xt = limbs.copy()
                    xt[:3] = base[:3]
                    xt[ri] = self.near_rotvec(Rotation.from_matrix(R).as_rotvec(), w0)
                    if raw:
                        xt[:3] += c0 - middle(xt)            # the root offset moves the whole body (armature space)
                    xt = np.clip(xt, lo, hi)
                    raw.append((self.outline_iou(t, self.full(xt)), xt))
        order = [0] + [k for k in sorted(range(len(raw)), key=lambda k: -raw[k][0]) if k][:int(self.w['rescue_keep'])]
        opts = []
        for k in order:
            xr = self.fit_outline(t, raw[k][1], rounds=3, iters=4)
            opts.append((xr, self.outline_iou(t, self.full(xr)), exp_so3(xr[ri]), raw[k][0]))
        return opts

    def rescue_orientation(self, x, log):
        """Frames where the warm start lost the outline (the tracks were wrong: a roll onto the back, a somersault, a
        spin) try the body turned over: half and quarter turns about its own three axes, each fitted to the outline
        alone, on every few frames of a run. A shortest path through the run picks one turn per key frame: outline
        overlap against the turns between neighbours (holding costs nothing), so a short slip never flips the body.
        The frames between follow from the key frame before, again by the outline. Returns the frames rescued."""
        ri = self.root_cols()
        if ri is None:
            return x, []
        T, w = self.T, self.w
        ious = np.array([self.outline_iou(t, self.full(x[t])) for t in range(T)])
        # with limb outlines, the trunk's alone decides: legs folded wrong lose the outline too, and need no turn
        doubt = np.array([self.trunk_iou(t, self.full(x[t])) for t in range(T)]) if self.plab is not None else ious
        n0 = max(3, T // 12)
        ref = float(np.median(doubt[:n0]))
        bad = [t for t in range(n0, T) if doubt[t] < ref * w['rescue_iou_share']]
        log(f'orientation: first frames cover the {"trunk" if self.plab is not None else "outline"} {ref:.2f}; '
            f'{len(bad)} frames lost it')
        if not bad:
            return x, []
        runs = [[bad[0]]]
        for t in bad[1:]:
            if t - runs[-1][-1] <= 3:                     # a frame or two back on the outline does not end a run
                runs[-1].extend(range(runs[-1][-1] + 1, t + 1))
            else:
                runs.append([t])
        short = [r for r in runs if len(r) < w['rescue_min_run'] and r[-1] < T - 1]
        if short:
            log(f'orientation: {len(short)} short runs back on the outline after them, kept')
        runs = [r for r in runs if r not in short]
        lo, hi = self.lo[self.act], self.hi[self.act]
        ang = lambda R1, R2: float(np.linalg.norm(Rotation.from_matrix(R1.T @ R2).as_rotvec()))
        step = int(w['rescue_step'])
        x = x.copy()
        candidates = lambda t: [o[:3] for o in self.turn_candidates(t, x[t])]
        rescued = []
        for run in runs:
            # a few frames first: a run where no turned body fits the outline better is left alone (most are: fast
            # moves and blur lose the outline too, and they need no turn)
            probe = sorted(set(run[::12] + [run[-1]]))
            cands = {t: candidates(t) for t in probe}
            gain = max(max((o[1] - c[0][1] for o in c[1:] if ang(o[2], c[0][2]) > .5), default=-1.0)
                       for c in cands.values())
            if gain < w['rescue_margin']:
                log(f'orientation: frames {run[0]}-{run[-1]}: no turn fits the outline better ({gain:+.2f}), kept')
                continue
            keys = run[::step] + ([run[-1]] if (len(run) - 1) % step else [])
            for t in keys:
                if t not in cands:
                    cands[t] = candidates(t)
            # shortest path over the key frames, from the last frame before the run
            a = run[0]
            prevR = exp_so3(x[a - 1][ri]) if a > 0 else None
            tc = w['rescue_turn_cost']
            cost = [[-o[1] + (tc * ang(prevR, o[2]) if prevR is not None else 0.0) for o in cands[keys[0]]]]
            back = [[0] * len(cands[keys[0]])]
            for i, t in enumerate(keys[1:], 1):
                pc = cands[keys[i - 1]]
                row, brow = [], []
                for o in cands[t]:
                    k = min(range(len(pc)), key=lambda k: cost[i - 1][k] + tc * ang(pc[k][2], o[2]))
                    row.append(cost[i - 1][k] + tc * ang(pc[k][2], o[2]) - o[1])
                    brow.append(k)
                cost.append(row)
                back.append(brow)
            k = int(np.argmin(cost[-1]))
            pick = {}
            for i in range(len(keys) - 1, -1, -1):
                pick[keys[i]] = cands[keys[i]][k][0]
                k = back[i][k]
            if all(ang(exp_so3(pick[t][ri]), exp_so3(x[t][ri])) < .5 for t in keys):
                log(f'orientation: frames {run[0]}-{run[-1]}: keeps its own turn')
                continue
            # the frames between key frames from the one before, fitted to the outline
            for t in run:
                if t in pick:
                    x[t] = pick[t]
                else:
                    guess = x[t].copy()
                    guess[ri] = x[t - 1][ri]
                    guess[:3] = x[t - 1][:3]
                    x[t] = self.fit_outline(t, np.clip(guess, lo, hi), prev=x[t - 1])
                rescued.append(t)
        for t in range(1, T):                               # rotation vectors continuous (smoothness compares numbers)
            x[t][ri] = self.near_rotvec(x[t][ri], x[t - 1][ri])
        if rescued:
            after = [self.outline_iou(t, self.full(x[t])) for t in rescued]
            log(f'orientation: {len(rescued)} frames turned over; they cover the outline {np.mean(after):.2f} '
                f'(were {np.mean(ious[rescued]):.2f})')
        return x, rescued

    def gate_tracks(self, x, frames):
        """In the given frames keep only the tracks whose point the pose can see: one behind the body (more than 3% of
        its size behind the surface at its pixel) is a tracker that slid onto another part."""
        if not hasattr(self, 'vis0'):
            self.vis0 = self.vis.copy()
        tol = .03 * self.height
        dropped = 0
        for t in frames:
            X = self.pose_all(self.full(x[t]))
            self.vis[:, t] = self.vis0[:, t]
            for vi, v in enumerate(self.views):
                ids = np.where((self.tview == vi) & (self.vis0[:, t] >= self.w['visible_min']))[0]
                if not len(ids):
                    continue
                tri, _, zbuf = self.cr.rasterize(project(v, X), view_depth(v, X), self.tris, v['size'])
                pts = np.einsum('km,kmd->kd', self.bary[ids], X[self.l_tv[ids]])
                pix = project(v, pts)
                d = view_depth(v, pts)
                W, H = v['size']
                c = np.clip(pix[:, 0].astype(int), 0, W - 1)
                r = np.clip(pix[:, 1].astype(int), 0, H - 1)
                hidden = (tri[r, c] >= 0) & (zbuf[r, c] < d - tol)
                self.vis[ids[hidden], t] = 0.0
                dropped += int(hidden.sum())
        return dropped

    # ----- solving -------------------------------------------------------------------------------------------
    def warm_start(self):
        """Frame by frame, each frame from the one before (a few damped Gauss-Newton steps)."""
        lo, hi = self.lo[self.act], self.hi[self.act]
        x = np.zeros((self.T, self.na))
        lo0, hi0 = self.start_bounds(lo, hi)
        x[0] = np.clip(self.p0[self.act], lo0, hi0)
        x[0] = self.lm_frame(0, x[0], None, lo0, hi0, iters=8)
        for t in range(1, self.T):
            guess = np.clip(x[t - 1] + (.5 * (x[t - 1] - x[t - 2]) if t > 1 else 0.0), lo, hi)
            if self.w['w_cover'] > 0:
                self.update_cover({t: guess}, [t])           # also into the silhouette it leaves uncovered
            x[t] = self.lm_frame(t, guess, x[t - 1], lo, hi, iters=8)
        return x

    def lm_frame(self, t, x, prev, lo, hi, iters):
        mu = 1e-2
        wt = self.smooth_w * .5                                   # stay near the previous frame
        robust = None

        def cost(xx):
            c = self.frame_cost(t, xx, robust)
            if prev is not None:
                c += np.sum((wt * (xx - prev)) ** 2)
            return c
        for it in range(iters):
            if it == 2:
                e = np.full(len(self.u), np.nan)
                p = self.full(x)
                X = np.zeros((len(self.rest), 3))
                X[self.tv_unique] = self.rig.skin(p, self.rest[self.tv_unique], self.idx[self.tv_unique],
                                                  self.wts[self.tv_unique])
                for vi, v in enumerate(self.views):
                    ids = np.where(self.tview == vi)[0]
                    pts = np.einsum('km,kmd->kd', self.bary[ids], X[self.l_tv[ids]])
                    e[ids] = np.linalg.norm(project(v, pts) - self.u[ids, t], axis=1)
                robust = self.robust_weights(e)
            H, g, c0 = self.normal(t, x, robust)
            if prev is not None:
                H += np.diag(wt ** 2)
                g += wt ** 2 * (x - prev)
                c0 += np.sum((wt * (x - prev)) ** 2)
            trials = []
            for k in range(6):
                step = np.linalg.solve(H + mu * 4 ** k * (np.diag(np.diag(H)) + 1e-6 * np.eye(self.na)), -g)
                trials.append(np.clip(x + step, lo, hi))
            if self.gpu is not None:                       # all six trial steps scored in one pass on the card
                cs = self.gpu.frames([t] * 6, np.stack(trials), None if robust is None else np.stack([robust] * 6),
                                     want_jac=False)
                if prev is not None:
                    cs = cs + np.sum((wt * (np.stack(trials) - prev)) ** 2, axis=1)
                better = [k for k in range(6) if cs[k] < c0]
            else:
                better = []
                for k in range(6):
                    if cost(trials[k]) < c0:
                        better = [k]
                        break
            if not better:
                break
            k = better[0]
            x, mu = trials[k], max(mu * 4 ** k / 3, 1e-6)
        return x

    def start_bounds(self, lo, hi):
        """Frame 0 is our own start render: its pose is known, so it may move only within pin_start (degrees; the
        root's offset within 1% of the height). 0 = free, as any other frame."""
        pin = self.w['pin_start_deg']
        if pin <= 0:
            return lo, hi
        p0 = self.p0[self.act]
        tol = np.where(np.asarray(self.act) < 3, .01 * self.height, np.radians(pin))
        return np.maximum(lo, np.minimum(p0 - tol, hi)), np.minimum(hi, np.maximum(p0 + tol, lo))

    def solve_clip(self, x, rounds, iters, log):
        lo = np.tile(self.lo[self.act], self.T)
        hi = np.tile(self.hi[self.act], self.T)
        lo[:self.na], hi[:self.na] = self.start_bounds(lo[:self.na], hi[:self.na])
        DtD = (self.D.T @ self.D).tocsr()
        for rnd in range(rounds):
            if self.w['w_cover'] > 0:
                n = self.update_cover(x)
                log(f'round {rnd}: {n} uncovered silhouette points matched')
            if rnd and getattr(self, 'rescued', None):
                self.gate_tracks(x, self.rescued)            # the turned frames' visible points, for the pose now
            E = self.track_errors(x)
            # A track that disagrees with the rig over many frames (it jumped to another part) loses weight.
            med = np.nanmedian(np.where(np.isnan(E), np.nan, E), axis=1)
            gnc = self.w['robust_start'] ** ((rounds - 1 - rnd) / max(rounds - 1, 1))
            self.rel = 1 / (1 + (np.nan_to_num(med, nan=0.0) / (2 * self.w['cauchy_px'] * gnc)) ** 2)
            robust = [self.robust_weights(E[:, t], gnc) for t in range(self.T)]
            mu = 1e-2

            frames = list(range(self.T))

            def total(xx):
                if self.gpu is not None:
                    c = float(self.gpu.frames(frames, xx, np.stack(robust), want_jac=False).sum())
                else:
                    c = sum(self.frame_cost(t, xx[t], robust[t]) for t in frames)
                s = self.D @ xx.ravel()
                return c + s @ s
            c0 = total(x)
            for it in range(iters):
                if self.gpu is not None:
                    Hb, gb, _ = self.gpu.frames(frames, x, np.stack(robust))
                    blocks, g = list(Hb), gb.ravel().copy()
                else:
                    blocks, g = [], np.zeros(self.T * self.na)
                    for t in frames:
                        Ht, gt, _ = self.normal(t, x[t], robust[t])
                        blocks.append(Ht)
                        g[t * self.na:(t + 1) * self.na] = gt
                H = sp.block_diag(blocks, format='csr') + DtD
                g += DtD @ x.ravel()
                dH = H.diagonal()
                accepted = False
                for _ in range(6):
                    A_ = H + sp.diags(mu * dH + 1e-6)
                    step = spsolve(A_.tocsc(), -g)
                    xn = np.clip(x.ravel() + step, lo, hi).reshape(x.shape)
                    cn = total(xn)
                    if cn < c0:
                        x, c0, mu, accepted = xn, cn, max(mu / 3, 1e-7), True
                        break
                    mu *= 4
                log(f'round {rnd} iter {it}: cost {c0:.0f} mu {mu:.1e}' + ('' if accepted else ' (no step)'))
                if not accepted:
                    break
        return x


def despike(x, act, rad, metres, passes=3):
    """One-frame jumps out and back (a blurred or dusty frame pulled a joint away): where a channel leaves its
    neighbours' mean by more than the threshold, and bends the other way than both neighbours do, it takes that mean.
    The top of a real fast swing bends the same way as the frames around it, so it is left alone. Returns (x, how many
    values changed)."""
    x = np.array(x, float)
    thr = np.where(np.asarray(act) < 3, metres, rad)
    changed = 0
    for _ in range(passes):
        mid = .5 * (x[:-2] + x[2:])
        dev = x[1:-1] - mid                               # (T-2): frame t's bend against its neighbours
        spike = np.zeros_like(dev, bool)
        s = np.sign(dev)
        spike[1:-1] = (np.abs(dev[1:-1]) > thr[None]) & (s[:-2] == -s[1:-1]) & (s[2:] == -s[1:-1])
        if not spike.any():
            break
        inner = x[1:-1]
        inner[spike] = mid[spike]
        changed += int(spike.sum())
    return x, changed


def solve(creature, clip, cfg, tracks, tag='-v2r', init=None, guide=None):
    """init: parameters [T, n] of an earlier pass to start from (no warm start). guide: parameters [T, n] of another
    tool's reading of the same clip (a learned motion model): every joint is pulled toward it (w_guide per radian;
    the root's offset is not, such tools do not place the body), and the fit starts from it (guide_init)."""
    t0 = time.time()
    if guide is not None and 'w_tip' not in cfg.get('solve', {}):
        # the limb's end at the far end of its SAM 2 outline: once that outline lost a leg (a plain gallop) it pulled
        # the paw and dragged the chest 30 degrees nose down; a guide already says where the limbs end. On the
        # bench's gallop the guided fit went from 17.3 to 12.5 degrees per bone without it
        cfg = dict(cfg, solve=dict(cfg.get('solve', {}), w_tip=0.0))
    pb = Problem(creature, clip, cfg, tracks)
    log = lambda s: print(f'[{time.time() - t0:6.1f}s] {s}', flush=True)
    log(f'{clip}: {pb.T} frames, {pb.na} active parameters, {len(pb.u)} tracks, {len(pb.verts)} vertices')
    if guide is not None:
        g = np.asarray(guide, float)
        if len(g) != pb.T:
            raise SystemExit(f'guide has {len(g)} frames, the clip {pb.T}')
        g = g[:, pb.act]
        for t in range(1, pb.T):                          # the root's rotation vector continuous, as the fit keeps it
            ri = pb.root_cols()
            if ri is not None:
                g[t, ri] = pb.near_rotvec(g[t, ri], g[t - 1, ri])
        pb.gait_target = g
        pb.gait_w = np.full((pb.T, pb.na), float(pb.w['w_guide']))
        pb.gait_w[:, np.asarray(pb.act) < 3] = 0.0
        # twist about a bone and sideways swing: the video barely shows them, the outline pulled them to their limits
        # (a hind paw turned 30 degrees about itself while the guide kept it straight); they follow the guide firmly
        role = getattr(creature, 'role', {})
        for i, c in enumerate(pb.act):
            r_ = role.get(creature.rig.channels[c - 3]) if 3 <= c < 3 + len(creature.rig.channels) else None
            if r_ == 'twist':
                pb.gait_w[:, i] *= float(pb.w['guide_offaxis'])
            elif r_ == 'side':                            # the front panel sees a leg come in under the belly
                pb.gait_w[:, i] *= float(pb.w['guide_side'])
        # the whole body's turn is ours: such a tool does not know the camera (it reared a galloping wolf up, and
        # pulled toward that the fit rolled the body over), while the trunk's tracks and outline place it well
        ri = pb.root_cols()
        if ri is not None:
            pb.gait_w[:, ri] *= float(pb.w['guide_root'])
        pb.w['mirror_legs'] = False                       # the guide already says what a poorly seen leg does
        if init is None and pb.w['guide_init'] in (True, 'copy'):
            x0 = np.clip(g, pb.lo[pb.act], pb.hi[pb.act])
            x0[:, np.asarray(pb.act) < 3] = pb.p0[pb.act][np.asarray(pb.act) < 3]
            if ri is not None and pb.w['guide_root'] < 1:     # the body as at the start; the tracks turn it
                x0[:, ri] = pb.p0[pb.act][ri]
            log(f'guide: every joint pulled toward another tool\'s reading (w_guide {pb.w["w_guide"]}), fit starts '
                f'from it')
            full = np.tile(pb.p0, (pb.T, 1))
            full[:, pb.act] = x0
            init = full
    if pb.plab is not None:
        log(f'parts: every limb followed by {"its colour" if pb.colour_parts else "SAM 2"}; {pb.parts_dropped} track '
            f'points that sat on another limb set aside')
        if pb.colour_parts:
            log('colours: ' + json.dumps(pb.colour_info['bones']))
    if len(pb.kp_v):
        n = sum(len(f) for fr in pb.kp_frames for f in fr)
        log(f'keypoints: {n} elbow, knee and paw points from the animal pose model; its left and right were swapped '
            f'in {pb.kp_swaps} leg pairs (frame and view), taken from the leg outlines instead')
    if init is None:
        x = pb.warm_start()
        log('warm start done')
        if pb.w['rescue']:
            x, pb.rescued = pb.rescue_orientation(x, log)
            if pb.rescued:
                log(f'orientation: {pb.gate_tracks(x, pb.rescued)} track points the turned body cannot see set aside')
    else:
        x = np.clip(np.asarray(init)[:, pb.act], pb.lo[pb.act], pb.hi[pb.act])
    x = pb.solve_clip(x, pb.w['rounds'], pb.w['lm_iters'], log)
    if pb.w['mirror_legs'] and pb.mirror_legs(x, log):
        x = np.clip(x, pb.lo[pb.act], pb.hi[pb.act])
        x = pb.solve_clip(x, max(2, pb.w['rounds'] - 1), pb.w['lm_iters'], log)
    if pb.w['despike_deg'] > 0:
        x, n = despike(x, pb.act, np.radians(pb.w['despike_deg']), .02 * pb.height)
        log(f'despike: {n} one-frame jumps set to their neighbours\' mean')
    fps = 1.0 / max(float(np.median(np.diff(pb.times))), 1e-6)
    if pb.w['lowpass_hz'] > 0 and pb.T > 12 and pb.w['lowpass_hz'] < .5 * fps:
        # as mocap is cleaned: a frame-to-frame shiver (a dusty frame's chest dipping for one frame and back) goes,
        # the stride stays (on the bench the error moved by under half a degree at 8 Hz, the shiver did not)
        from scipy.signal import butter, filtfilt
        b, a = butter(2, pb.w['lowpass_hz'] / (.5 * fps))
        x = filtfilt(b, a, x, axis=0, padlen=min(8, pb.T - 2))
        log(f'low-pass: {pb.w["lowpass_hz"]:.0f} Hz on {fps:.0f} fps content')
    x = np.clip(x, pb.lo[pb.act], pb.hi[pb.act])
    x[0] = np.clip(x[0], *pb.start_bounds(pb.lo[pb.act], pb.hi[pb.act]))
    P = np.stack([pb.full(x[t]) for t in range(pb.T)])
    E = pb.track_errors(x)
    meta = {'tool': 'video2rig', 'weights': pb.w, 'tracks': int(len(pb.u)),
            'track_error_median_px': float(np.nanmedian(E)), 'seconds': round(time.time() - t0, 1)}
    path = creature.save(clip, P, tag, json.dumps(meta))
    log(f'saved {path} (track error median {meta["track_error_median_px"]:.2f} px)')
    return path
