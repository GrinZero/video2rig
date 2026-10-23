"""The rig onto the reader's joints: frame by frame, the pose whose keypoints land where the reader saw them in every
panel (each point weighted by how sure the reader was, far misses barely pulling), within the rig's joint ranges,
starting from the frame before (frame 0 from the start pose, which the clip's first frame shows). Then the mocap
clean-up of the tool (one-frame spikes, an 8 Hz low-pass) and the motion file for blender/import_motion.py.

    python -m video2rig.reader.fit <project> <clip> [--tag -reader]
        -> work/<clip>/fit<tag>.npz (as the tool's fits: review, score) and out/<clip><tag>.npz (Blender)

Nothing here knows any motion: no gait, no mirror twin, no guide. Where the reader is unsure in both panels, the
joint stays near the frame before; that is reported.
"""
import argparse
import json
import re
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from ..common import Project, write_json
from ..creature import Creature
from ..solve import despike
from . import KeypointRig, along, end_pairs, is_offaxis, side_of, twin_label

SETTINGS = {
    'loss': 'cauchy',          # far misses barely pull (a wrong reading); a limb the frame before got wrong is caught
                               # by the second start instead (two_starts). soft_l1 (far misses pull) gave fewer swaps
                               # but let wrong readings drag the body: bench gallop 8.5 deg against 7.6
    'cauchy_px': 6.0,          # robust scale of a keypoint's miss
    'stable_scale': 2.0,       # and of a sure reading that moves smoothly (conf over stable_conf in the frame and both
    'stable_conf': .6,         # around it, bent by under stable_px of the height between them), times this: a far
    'stable_px': .02,          # leg's sure front-panel reading kept pulling while the hidden leg drifted. Bench 2.99 ->
                               # 2.90 deg at 1.5-3 alike; 1.75-2 also calmer real clips, 1.5 and 2.5-3 threw an alien's
                               # spin (combo01) into whole-body flips (1 = off)
    'two_starts': True,        # also start each frame from the frame before with every limb in its start pose: a
                               # galloping wolf's far hind leg stayed stretched back while the reader saw it under the
                               # belly (run01, frame 55)
    'read_start': True,        # also start each frame from the limbs set where both panels read them: a man dropping
                               # into a squat (knee 50 deg in 3 frames) kept his right leg straight and back for 40
                               # frames, the start-pose start winning a frame by a hair and the frames after following
    'conf_floor': .05,         # a reading below this peak height counts for nothing
    'w_smooth': 30.0,          # px-equivalent per radian (and per metre / 20) toward the frame before
    'w_prior': 2.0,            # px-equivalent per radian toward the start pose (the tool's prior weights x this / 2)
    'w_floor': 3000.0,         # px-equivalent per metre: no keypoint under the floor (a galloping paw went 3.5 cm in)
    'despike_deg': 6.0,
    'turn_quat': True,         # the body's turn cleaned up as a turn (smooth_turn), not as three numbers
    'despike_keep_px': 1.0,    # a "spike" the reading backs (its pose lands this much closer, median px) stays: at 24 fps
                               # a galloping wolf's front paw really turns back 15-20 deg in one frame, and smoothing
                               # 287 such frames left its front legs 27 px from the reading, stiff (18 px without)
    'despike_limbs': True,     # and judged per limb (its own points read surer than .3), not only over the whole body
    'lowpass_hz': 8.0,         # bones that move fast (a galloping paw: 260-360 deg/s) keep their snap
    'lowpass_slow_hz': 4.0,    # bones that move slower than slow_deg_s: the reading's noise shows on them as a tremble
    'slow_deg_s': 150.0,       # (the wyvern's wing roots and hands jolted 2-9 deg a frame while turning ~3). The reader
                               # reads every frame on its own and shakes most on the small parts (a centaur's elbow 10.7
                               # px from frame to frame where both panels see the same height, its spine 0.7-1.5 px),
                               # read sure. 5 -> 4 Hz (with burst_deg_s): the limbs' jolt 1.75 -> 1.61% of the height
                               # over 43 real clips, arms calmer when still in 30, the bench 6.77 -> 6.78 deg; 3 Hz eats
                               # strikes (sure readings in fast frames missed 8.5 -> 10.0%, bench 7.09 deg). A cutoff
                               # per frame from each bone's speed, a robust spline and keypoints weighted by their
                               # shake were all worse (dev/SHAKE-2026-10-18.md)
    'burst_deg_s': 450.0,      # a slow bone that still turns faster than this somewhere (on its 3 Hz path) keeps
    'lowpass_burst_hz': 5.0,   # lowpass_burst_hz: an alien's legs, still most of its jump attack, fold for the landing
                               # in 2-3 frames, and at 4 Hz the knees bent too little there (combo01, 132-134)
    'slow_p90_deg_s': 450.0,   # but not a bone with fast moments (90th percentile over this): the boss's arm, still most of
                               # the clip, flaps back along its flank for 2 frames in every 6; at 5 Hz the flaps kept 60%
                               # of their reach, at 8 Hz 80%
    'twin_side': True,         # swap a side panel's near/far twin readings when the front panel agrees better
    'twin_jumps': True,        # swap a point read where its twin is for a frame back
    'twin_conf': .1,           # both twins' readings at least this sure for the side panel swap
    'twin_switch': 6.0,        # the side panel swap decided along the clip: frames of clear evidence a change of mind
                               # needs (0 = frame by frame); 1.5 swapped a wolf's walk in stretches of 5-7 frames
    'twin_margin': .04,        # clear evidence: swapped beats kept by this share of the height per point (in px of
                               # the side panel); was 4 px flat
    'twin_link': 0.0,          # a jump swap has to bring the point this much nearer the point before it on its limb (0 =
                               # off): made for a tracker's lunge (feet, read right, swapped for 23 frames). Off since
                               # 2026-10-19: humans are not this tool's job, and off it no creature clip got worse and
                               # three got better (dev/RULES-2026-10-19.md); .8 was the value
    'twin_same': True,         # a point read on top of its twin while the limbs are apart counts little: the reader took
    'twin_same_px': .06,       # one limb for the other (share of the height in the side panel's pixels)
    'twin_same_weight': .4,    # .15 cost the wolf bench's gallop 0.3 deg; .4 fixed the tracker's knees as well
    'twin_link_apart': True,   # the twin_link test only when the points before are apart: two feet side by side say
                               # nothing about which toes are whose (a tracker's crawl, side panel, frame 56: the toes
                               # read swapped, the swap refused because both feet stood in one spot)
    'bridge': True,            # a point read unsure (conf under bridge_conf, or on top of its twin) for a short stretch
    'bridge_conf': .4,         # between sure readings goes from one sure reading to the next in a straight line: a hand
    'bridge_max': 24,          # hidden behind the body was guessed 100-300 px apart frame after frame (conf .1-.26) and
                               # the arm shook (frames; longer stretches are left as read)
    'twin_lean': 1.0,          # an unclear frame counts as much for the reading's own labels as a clear one against
                               # them: a swap needs clear frames to outnumber unclear ones (an alien's run: 60% of the
                               # frames clearly swapped; a wolf's walk 25%, where Seedance drew another stride in the
                               # front panel); was .3
    'foot_lock': True,         # a foot that stands in the video stands in the motion (lock_feet): the last bone of a leg
    'lock_height': .03,        # the creature stands on, down (both ends within lock_height of the height above where
    'lock_speed': .2,          # they rest in the start pose) and still (under lock_speed heights a second, in the
    'lock_frames': 3,          # motion and in the video) for lock_frames, pinned where it stands, easing in and out over
    'lock_blend': 3,           # lock_blend frames. 25 creature clips (2026-10-19): the standing legs slide 281 -> 206
                               # (sum, % of the height; the troll's combo 281 -> 168, the wolf alpha's howl 37 -> 4), the
                               # bench 6.63 -> 6.30 deg, the regression gate 25 clips as approved. A walk Seedance draws
                               # on the spot (the body still, the feet sliding back in the video too) is not pinned
    'steps': True,             # a walk Seedance draws on the spot moves the body by its steps (root_from_steps), so the
    'steps_low': .04,          # standing feet stand and the body walks; lock_feet then pins them. 9 walk / run clips
    'steps_back': .08,         # of the laptop (2026-10-19): the standing legs' slide alien walk 161 -> 9, centaur walk
    'steps_share': .55,        # 655 -> 138, wolf alpha walk 552 -> 282, alien run 632 -> 207; fights untouched. The body
    'steps_same_way': .92,     # then travels where the video stays: the panels' overlay drifts in review films, by
    'steps_hz': 2.0,           # intent (his rule: nothing moves without the legs moving it)
    'steps_steady': .5,        # and the pace steady: the spread of the sliding feet's speed (std / mean) at most this.
                               # A walk's steps slide back evenly (walks and runs 0.13-0.53), a fight's feet in jolts
                               # (0.60-1.06): with the share alone at .5, a centaur's attack (51%, 0.95 one way) was
                               # taken for a walk and moved 1.4 heights out of its panels (2026-10-19)
    'springs': True,           # tails and ears follow through a damped spring (spring_pass): their jolt -18% over 12
    'spring_hz': 4.0,          # clips with them, none worse (2026-10-19); not wings, whose flap broke
    'spring_zeta': .55,
    'spring_soft': .4,
    'no_self_pass': True,      # a free limb's points (hands, wrists, elbows) stay out of the creature's own trunk:
    'self_core': 1.0,          # capsules round the trunk's bones, their radius self_core x the median distance of the
    'self_w': 3000.0,          # bone's own skin from it (inside the body, so a hand may lie on the chest), px-equivalent
                               # per height of depth in. A hand one camera cannot see (a centaur's sword hand, behind the
                               # body in the side panel) went into the horse's flank (2026-10-19)
    'floor_bend': 'still',     # a limb end under the floor after the clean-up bends out of it by its own limb: 'still',
                               # the least change to what the cameras see of the low-passed pose; 'read', toward that
                               # frame's readings (before 2026-10-18), which put the reading's frame-to-frame shake
                               # back into every such frame (the limbs' jolt 2.02 -> 1.75% of the height over 43
                               # real clips, 9 calmer, none worse; miss and bench unchanged)
    'flat_feet': 120.0,        # px-equivalent per unit of a planted foot's tilt (sin): the keypoints run along a foot,
                               # nothing read says how it rolls about them; an alien's planted feet rolled 40-56 deg
                               # with every point on its reading (0 = off)
    'turn_starts': True,       # a frame fitted badly (median miss over turn_miss_px) is also started with the body
    'turn_miss_px': 8.0,       # turned about the up axis (45, 90, 180 deg): an alien's spin stayed within 50 deg of
                               # the frame before, every start began there
    'hidden_refit': False,     # fit again with the readings of points the fitted body hides in a panel weighted down
    'hidden_weight': .1,       # (off: untested on the clips that are right; built for a hand read on a spine of the
                               # back, which turned out to be the side panel's arms mixed up)
    'panel_drift': False,      # off since 2026-10-19: it fired on the troll's combo01 (panels 5.5% apart) and threw its
                               # arms off for a second; off, no creature clip got worse (dev/RULES-2026-10-19.md).
                               # Two orthographic panels both see one axis (side | top: along the body; front | side: up),
    'drift_min': .05,          # and Seedance does not always move the body the same in both: the fps-range boss's charge
                               # slid 2x as far in the side panel as in the top one (30-70 px apart). Per frame the
                               # centre line's offset along that axis, split half to each panel, so every limb is judged
                               # against the same body; only in a clip where it is over drift_min of the height (median):
                               # the bench clips, rendered through the cameras, read 2-5 px of it, noise
    'panel_pick': True,        # which panel to believe, per limb point and frame: a panel that sees a limb from behind
    'pick_share': .12,         # the body guesses it (the boss's right arm, swept back along its flank in the top panel,
    'pick_weight': .1,         # was read forward, on the left one, in the side panel: the fit split the difference and
                               # the arm hung still). Where the panels disagree on such a point (pick_panels), its reading
                               # in that panel weighs pick_weight
    'read_one_panel': True,    # the read start also sets a limb point only one panel reads, its depth from the frame
                               # before: after the pick, a far arm has one panel left
    'flip': False,             # a limb piece read turned round in a frame, its upper joint still, and back a few frames
    'flip_deg': 60.0,          # later counts flip_weight in between (flipped()). Off: a sword strike or a backhand is
    'flip_back_deg': 30.0,     # exactly that (forearm and hand whip round past a still elbow and back) and was smoothed
                               # away (centaur attack01 31-56, troll combo01 71-91); the "calmer" real runs (troll
                               # throw01 112 -> 16 jumps) were strikes erased, which a jump count cannot tell
    'flip_weight': .1,
    'twin_jump': 0.0,          # side_path: no change of mind that puts more than this many twin_sigma of jump per point
                               # into the readings (two limbs read apart and smooth, swapped there, would teleport; 0 =
                               # off). Off: at 0.5 it moved a centaur's arm swap (attack01, 68-92 -> 59-120) and the
                               # sword arm, given the other arm's readings, twisted behind the head; it had fixed a
                               # tracker's turn (turns02 47-68) and real clips' pops 692 -> 598
    'depth_w': .2,             # a depth reader's depth (stage R: each keypoint along the panel's line of sight, from the
                               # body's middle) as one more measurement, in px-equivalents of that panel times this: it
                               # reads depth to 1-2.5 cm, the other panel to ~5 mm, so it speaks where a panel does not see
    'depth_where': 'unseen',   # 'unseen': a panel's depth of a joint counts only as much as the other panel does NOT
                               # read that joint (1 - its confidence): the camera that sees a joint plainly decides; 'all'
    'who_sees': False,         # who sees, decides (plain_readings): a joint a panel reads surely and smoothly is a
    'who_sees_parts': 'veto,model',  # fact in that panel, and the left / right check (fix_twins) may not hand it to the
                               # other limb in the middle of its track. veto (no_cut_tracks): a swap of side_path's
                               # that begins or ends in the middle of such a track is given back to the reader's
                               # own labels; it only takes swaps away (0 of 41 real clips swap more, 4.4% -> 1.6%
                               # of the plain readings moved before the fit, a still joint drifting over 5% of
                               # the height in 12 -> 7 clips; bench 2.90 -> 2.89 deg, injected swaps 70 -> 71
                               # jumps). model (fit_clip, body_picture): where the panels put a joint at different
                               # heights, a reading that lies on another part of the body yields to one that lies
                               # on the joint's own limb. twoway, off: a reading put back on its limb by the jump
                               # check has to be one played backwards too (a troll's throw better, a centaur's
                               # strike worse). Off until the films were checked by eye (dev/bench/seen.py)
    'twin_mode': 'clear',      # the side panel's twins: 'path' (side_path: a cost per frame, a cost per change of mind,
    'twin_sigma': .03,         # frame 0 as read) or 'clear' (the clear / unclear evidence and twin_switch above). Faults
    'twin_cap': 3.0,           # injected on the bench (a limb swapped with its twin for 2-6 frames in 15% of the frames):
    'twin_cost': 4.0,          # 'clear' left them as read (550 jumps, 3.81 deg), 'path' 67 jumps, 3.60; clean 3.48 / 3.47.
                               # But on real clips 'path' made 11 creature clips worse than 'clear' (the troll's combo01
                               # arms, its throw, the wolf alpha's walk, the wyvern's fire for 57 frames) and helped one
                               # (an alien's run, 4 frames); 'clear' since 2026-10-19 (dev/RULES-2026-10-19.md)
}


def creature_for(pr, cfg, kps):
    """The clip's creature, with the bones the reader sees at both ends but the tool does not fit by default (a
    jaw: a bite opens it) added to the fitted ones. Returns the creature and {channel: (lo, hi)} ranges for them:
    the jaw opens one way, from shut to 40 degrees, and does not twist or swing sideways."""
    cr = Creature(pr, cfg)
    rig = cr.rig
    ends = {}
    for kp in kps:
        _, bone, f = kp
        if str(bone).startswith('@'):
            continue
        ends.setdefault(bone, set()).add(is_offaxis(kp) or f > 0)
    extra = [b for b, e in ends.items() if e == {False, True} and rig.names.index(b) not in rig.chan_of
             and not (rig.desc[rig.names.index(b), rig.root] and rig.names.index(b) != rig.root)]
    if not extra:
        return cr, {}
    fitted = list(dict.fromkeys(rig.names[j] for j, _ in rig.channels)) + extra
    cfg = {**cfg, 'fit_bones': fitted}
    cr = Creature(pr, cfg)
    rig = cr.rig
    p0 = cr.p0(None)
    narrow = {}
    for b in extra:
        j = rig.names.index(b)
        tip0 = rig.bone_points(p0)[1][j]
        best = None
        for a in (0, 1, 2):
            q = p0.copy()
            q[rig.index(b, a)] += .3
            down = float(-(rig.bone_points(q)[1][j] - tip0) @ cr.up)
            if best is None or abs(down) > abs(best[1]):
                best = (a, down)
        for a in (0, 1, 2):
            c = rig.index(b, a)
            if a == best[0]:
                narrow[c] = (p0[c] - .05, p0[c] + .7) if best[1] > 0 else (p0[c] - .7, p0[c] + .05)
            else:
                narrow[c] = (p0[c] - .08, p0[c] + .08)
    return cr, narrow


def fix_twins(cr, clip, uv, conf, kps, log=print, s=SETTINGS):
    """Left and right checked, not guessed: a side panel can mistake the near leg for the far one, the front panel
    sees them apart. Per frame and per pair of twin legs, the side panel's readings of the two legs are kept or
    swapped, whichever lets every point of both legs land on one 3D point with its front reading (orthographic
    cameras: the residual of the least-squares point from both panels). Swapped only when clearly better."""
    views = cr.views(clip)
    if len(views) < 2 or any(not hasattr(v['cam'], 'A') or v['cam'].type != 'ortho' for v in views):
        return uv, {}
    A = np.concatenate([v['cam'].A for v in views], 0)                  # (2V, 3)
    b = np.concatenate([v['cam'].b for v in views], 0)
    H = A @ np.linalg.pinv(A) - np.eye(len(A))                          # residual of the best point
    labels = [k[0] for k in kps]
    # the panel that may confuse near and far: the one looking across the creature (its axis most along the creature's
    # left-right; "least along the facing" took the view from above, which sees the twins apart, for a side | top clip)
    lateral = np.cross(cr.up, cr.facing)
    side = int(np.argmax([abs((v['cam'].fwd @ v['cam'].Ma) @ lateral) for v in views]))
    rig = cr.rig
    chains = cr.limb_chains(as_list=True)                               # limbs by the skeleton's shape
    first = {rig.names[ch[0]]: ci for ci, ch in enumerate(chains)}
    chain_of = {rig.names[j]: ci for ci, ch in enumerate(chains) for j in ch}
    pairs = {}
    for ci, ch in enumerate(chains):
        n0 = rig.names[ch[0]]
        tw = twin_label(n0)
        if tw not in first or side_of(n0) != 'L':
            continue
        ks = [(k, labels.index(twin_label(lb))) for k, lb in enumerate(labels)
              if chain_of.get(kps[k][1]) == ci and twin_label(lb) in labels]
        if ks:
            pairs[n0] = ks
    uv = uv.copy()
    swaps = {key: 0 for key in pairs}
    # a frame's evidence has to beat what the panels disagree by anyway: Seedance does not always draw the same
    # stride in both panels (a wolf's walk: the side panel swings the near foreleg, the front panel lifts the far one
    # in the same frames, 3-4% of the height apart per point; an alien's mixed-up arms were 8%)
    margin = float(s.get('twin_margin', .05)) * cr.height * float(np.linalg.norm(views[side]['cam'].A[1]))

    def resid(points):                                                  # points (n, 2V) -> summed residual, px
        return float(np.linalg.norm((points - b) @ H.T, axis=1).sum())
    # per frame the evidence (clearly better swapped, clearly better kept, or unclear), then one path through the
    # clip (Viterbi) that changes its mind only for twin_switch frames of evidence: a reader that mixes the side
    # panel's twins up does it for a stretch (an alien's run, its arms from the first stride on), and judged frame by
    # frame the unclear frames between kept the mix-up, the arm jumping back and forth in the first two strides
    height_px = cr.height * float(np.linalg.norm(views[side]['cam'].A[1]))
    # who sees, decides: what each panel plainly sees, taken from the readings as the reader gave them
    parts = set(str(s.get('who_sees_parts', '')).split(',')) if s.get('who_sees', False) else set()
    plain = plain_readings(cr, clip, uv, conf, s) if parts else None
    for key, ks in (pairs.items() if s.get('twin_side', True) else []):
        if s.get('twin_mode', 'path') == 'path':
            state = side_path(uv, conf, ks, side, H, b, height_px, s)
            if 'veto' in parts:
                state = no_cut_tracks(state, uv, plain, ks, side, float(s.get('stable_px', .02)) * height_px)
            for t in np.nonzero(state)[0]:
                for l, r in ks:
                    uv[t, side, [l, r]] = uv[t, side, [r, l]]
                    conf[t, side, [l, r]] = conf[t, side, [r, l]]
            swaps[key] = int(state.sum())
            continue
        ev = np.zeros(len(uv), int)
        for t in range(len(uv)):
            w = np.array([min(conf[t, side, l], conf[t, side, r]) for l, r in ks]) > s.get('twin_conf', .1)
            if w.sum() < 2:
                continue
            keep = swap = 0.0
            for (l, r), ok in zip(ks, w):
                if not ok:
                    continue
                pl = np.concatenate([uv[t, v, l] for v in range(len(views))])
                pr = np.concatenate([uv[t, v, r] for v in range(len(views))])
                ql, qr = pl.copy(), pr.copy()
                ql[2 * side:2 * side + 2], qr[2 * side:2 * side + 2] = pr[2 * side:2 * side + 2], pl[2 * side:2 * side + 2]
                keep += resid(np.stack([pl, pr]))
                swap += resid(np.stack([ql, qr]))
            if swap < .6 * keep and keep - swap > margin * w.sum():
                ev[t] = 1
            elif keep < .6 * swap and swap - keep > margin * w.sum():
                ev[t] = -1
        jump = float(s.get('twin_switch', 6.0))
        # an unclear frame leans to the reading's own labels: a swap holds through gaps between clear frames (an
        # alien's run: 1-3 frames) where clear frames outnumber them, not through a clip with a few scattered ones (a
        # centaur's run: 9 of 121) or a wolf's walk where the front panel lifts the other foreleg now and then
        cost = np.stack([(ev == 1).astype(float), (ev == -1) + float(s.get('twin_lean', 1.0)) * (ev == 0)], 1)
        acc, back = cost[0] + [0.0, jump], np.zeros((len(uv), 2), int)      # the reading's own labels to start
        for t in range(1, len(uv)):
            new = np.zeros(2)
            for st in (0, 1):
                stay, move = acc[st], acc[1 - st] + jump
                back[t, st] = st if stay <= move else 1 - st
                new[st] = min(stay, move) + cost[t, st]
            acc = new
        state = np.zeros(len(uv), int)
        state[-1] = int(np.argmin(acc))
        for t in range(len(uv) - 1, 0, -1):
            state[t - 1] = back[t, state[t]]
        for t in np.nonzero(state)[0]:
            for l, r in ks:
                uv[t, side, [l, r]] = uv[t, side, [r, l]]
                conf[t, side, [l, r]] = conf[t, side, [r, l]]
        swaps[key] = int(state.sum())
    log(f'twins: side panel readings swapped back to their own leg in {swaps} frames')
    # and in every panel, a point read where its twin is for a frame or a few (an alien's feet on the floor side by
    # side: the side panel read the near foot and toes where the far ones were and back, 130-150 px, often the foot
    # alone or the foot and toes, not the shin; the fit followed and the leg kicked out of the walk's rhythm): each
    # point is predicted from its last two readings (constant speed, so a gallop's paws passing each other stay
    # theirs) and swapped with its twin when the twin's prediction fits both clearly better; only points the panel
    # shows apart from their twin
    jumps = {key: 0 for key in pairs}
    link_apart = float(s.get('twin_same_px', .06)) * cr.height * float(np.linalg.norm(views[side]['cam'].A[1]))

    def jump_pass(uv, conf):
        """The frames in the order given, each judged against the ones before it as corrected so far:
        (T, V, K) True at the first of a pair of twins swapped there. Works on copies."""
        uv, conf = uv.copy(), conf.copy()
        done = np.zeros(conf.shape, bool)
        for t in range(1, len(uv)):
            for vi in range(len(views)):
                for key, ks in pairs.items():
                    for i, (l, r) in enumerate(ks):
                        if min(conf[t, vi, l], conf[t, vi, r], conf[t - 1, vi, l], conf[t - 1, vi, r]) <= .2:
                            continue
                        if min(np.linalg.norm(uv[t - 1, vi, l] - uv[t - 1, vi, r]),
                               np.linalg.norm(uv[t, vi, l] - uv[t, vi, r])) < 20.0:
                            continue
                        if t >= 2:
                            pl = 2 * uv[t - 1, vi, l] - uv[t - 2, vi, l]
                            pr = 2 * uv[t - 1, vi, r] - uv[t - 2, vi, r]
                        else:
                            pl, pr = uv[t - 1, vi, l], uv[t - 1, vi, r]
                        keep = np.linalg.norm(uv[t, vi, l] - pl) + np.linalg.norm(uv[t, vi, r] - pr)
                        swap = np.linalg.norm(uv[t, vi, l] - pr) + np.linalg.norm(uv[t, vi, r] - pl)
                        if i > 0 and float(s.get('twin_link', .8)) > 0:
                            # and the swap has to bring each point to its own limb: nearer the point before it on
                            # the chain (a foot to its shin). A lunging run (a tracker's scuttle) broke the
                            # constant-speed guess: the side panel's feet, read right, were swapped from frame 90
                            # on and stayed swapped for 23 frames (each frame judged against the swapped one
                            # before), the legs then fitted between both readings, one into the other
                            ul, ur = uv[t, vi, ks[i - 1][0]], uv[t, vi, ks[i - 1][1]]
                            link_keep = np.linalg.norm(uv[t, vi, l] - ul) + np.linalg.norm(uv[t, vi, r] - ur)
                            link_swap = np.linalg.norm(uv[t, vi, r] - ul) + np.linalg.norm(uv[t, vi, l] - ur)
                            told = not s.get('twin_link_apart', True) or np.linalg.norm(ul - ur) >= link_apart
                            if told and link_swap > float(s.get('twin_link', .8)) * link_keep:
                                continue
                        if swap < .3 * keep and keep - swap > 40.0:
                            uv[t, vi, [l, r]] = uv[t, vi, [r, l]]
                            conf[t, vi, [l, r]] = conf[t, vi, [r, l]]
                            done[t, vi, l] = True
        return done
    if s.get('twin_jumps', True):
        done = jump_pass(uv, conf)
        if 'twoway' in parts:
            # who sees, decides: a reading put back on its own limb has to be one whichever way the film is
            # played. This check follows each point from the frame before, and one wrong swap then carried itself
            # on for as long as the twins stayed apart: a troll's feet, their labels flickering in the front panel
            # while a stone covered them (throw01, 93-99), stayed swapped through the 40 frames after, both read
            # plainly and right, 200 px apart. Played backwards that swap is not there; a reading that really
            # sits on its twin for a few frames is found from both ends. Where the two disagree the reader's
            # own labels stand
            done &= jump_pass(uv[::-1], conf[::-1])[::-1]
        twin_of = {l: (r, key) for key, ks in pairs.items() for l, r in ks}
        for t, vi, l in zip(*np.nonzero(done)):
            r, key = twin_of[int(l)]
            uv[t, vi, [l, r]] = uv[t, vi, [r, l]]
            conf[t, vi, [l, r]] = conf[t, vi, [r, l]]
            jumps[key] += 1
    log(f'twins: readings that jumped to the other limb from one frame to the next, swapped back in {jumps} '
        f'panel-frames')
    if s.get('flip', False):
        log(f'flip: limb pieces read turned round for a few frames and back, counted little, in '
            f'{flipped(cr, clip, uv, conf, kps, s)} panel-points')
    # and a point read on top of its twin while the points next to them on the limbs are apart: the reader took one
    # limb for the other there (a tracker's lunge, side panel, frames 88-90: the far knee hidden and blurred, read 10 px
    # from the near one with the feet 110 px apart; the fit pulled both knees together). Both readings count little,
    # the other panel and the rest of the limbs place the joint
    amb = {key: 0 for key in pairs}
    near = float(s.get('twin_same_px', .06)) * cr.height * float(np.linalg.norm(views[side]['cam'].A[1]))
    for t in range(len(uv) if s.get('twin_same', True) else 0):
        for vi in range(len(views)):
            for key, ks in pairs.items():
                for i, (l, r) in enumerate(ks):
                    if np.linalg.norm(uv[t, vi, l] - uv[t, vi, r]) >= near:
                        continue
                    apart = [np.linalg.norm(uv[t, vi, ks[j][0]] - uv[t, vi, ks[j][1]]) for j in (i - 1, i + 1)
                             if 0 <= j < len(ks)]
                    if apart and max(apart) > 3 * near:
                        conf[t, vi, [l, r]] *= float(s.get('twin_same_weight', .4))
                        amb[key] += 1
    log(f'twins: readings on top of their twin with the limbs apart, counted little, in {amb} panel-points')
    if s.get('bridge', True):
        uv, n_br = bridge_unsure(uv, conf, s)
        log(f'bridge: unsure readings set between the sure ones around them in {n_br} panel-points')
    return uv, swaps


def flipped(cr, clip, uv, conf, kps, s=SETTINGS):
    """A limb piece read turned round in one frame, its upper joint still, and turned back a few frames later, was read
    on something else in between: a centaur's hind hoof, under its still ankle, read pointing 90 deg forward onto the
    front hoof for 6 frames and back (attack01, 19-24); the fit swung the leg there and back. Per panel and limb, a piece
    between two sure points (conf .3+) whose direction in the panel turns over flip_deg from one frame to the next
    while its upper point moves under 2% of the height, and comes back within flip_back_deg of where it was within
    bridge_max frames: its lower point and the points below it count flip_weight there (the bridge then sets them between
    the sure ones around). A turn that does not come back is left as read. Edits conf in place; returns panel-points."""
    views = cr.views(clip)
    rig = cr.rig
    turn, back = np.radians(float(s.get('flip_deg', 60.0))), np.radians(float(s.get('flip_back_deg', 30.0)))
    longest, weight = int(s.get('bridge_max', 24)), float(s.get('flip_weight', .1))
    T = len(uv)

    def angle(a, b):
        c = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)
        return np.arccos(np.clip(c, -1.0, 1.0))
    n = 0
    for ch in cr.limb_chains(as_list=True):
        order = [k for j in ch for _, k in sorted((along(kp, rig), k) for k, kp in enumerate(kps)
                                                   if not str(kp[1]).startswith('@') and kp[1] == rig.names[j])]
        for vi, v in enumerate(views):
            px = cr.height * v['cam'].px_per_unit
            for i in range(1, len(order)):
                p, k = order[i - 1], order[i]
                vec = uv[:, vi, k] - uv[:, vi, p]
                ok = (conf[:, vi, p] >= .3) & (conf[:, vi, k] >= .3) & (np.linalg.norm(vec, axis=-1) > .015 * px)
                last, t = None, 0
                while t < T:
                    if not ok[t]:
                        t += 1
                        continue
                    # against the last frame the piece was read surely (an unsure frame between says nothing)
                    if (last is None or t - last > longest or angle(vec[t], vec[last]) <= turn
                            or np.linalg.norm(uv[t, vi, p] - uv[last, vi, p]) >= .02 * px * (t - last)):
                        last, t = t, t + 1
                        continue
                    e = t + 1
                    while e < T and e - last <= longest and not (ok[e] and angle(vec[e], vec[last]) < back):
                        e += 1
                    if e >= T or e - last > longest:
                        last, t = t, t + 1
                        continue
                    for q in order[i:]:
                        conf[last + 1:e, vi, q] *= weight
                    n += e - last - 1
                    last, t = e, e + 1
    return n


def plain_readings(cr, clip, uv, conf, s=SETTINGS):
    """(T, V, K) True where a panel plainly sees a keypoint: the reader is sure of it (conf stable_conf or more) in
    the frame and in both around it, and the reading moves smoothly through them (bent by under stable_px of the
    creature's height). No guess and no jump onto another limb: what that camera shows of the joint is a fact."""
    views = cr.views(clip)
    sure = conf >= float(s.get('stable_conf', .6))
    acc = np.array([float(s.get('stable_px', .02)) * cr.height * v['cam'].px_per_unit for v in views])
    ix = np.clip(np.arange(len(uv)), 1, max(len(uv) - 2, 1))
    nx = np.minimum(ix + 1, len(uv) - 1)
    bend = np.linalg.norm(uv[ix - 1] - 2 * uv[ix] + uv[nx], axis=-1)
    return sure[ix - 1] & sure[ix] & sure[nx] & (bend < acc[None, :, None])


def no_cut_tracks(state, uv, plain, ks, side, apart):
    """side_path's swaps without those that cut a track the side panel plainly sees. A stretch of swapped frames
    begins and ends with a change of mind; where one of the two falls in the middle of a track a twin is read on
    surely and smoothly (plain in the frame before and in this one, further from the other twin's reading than
    `apart`, what a smooth track may bend), the reader did not jump from one limb to the other there, so the swap
    cannot begin or end there: that joint would teleport to the other limb's place (a troll landed on its hands,
    its two planted feet, read surely and still, swapped for 13 frames and back: the fitted feet changed places,
    combo01 46-58). Such a stretch is given back to the reader's own labels, whole. Only ever takes swaps away."""
    T = len(state)
    cut = np.zeros(T, bool)                                             # cut[t]: a change between t - 1 and t
    for l, r in ks:
        gap = np.linalg.norm(uv[:, side, l] - uv[:, side, r], axis=1)
        for k in (l, r):
            seen = plain[:, side, k] & (gap > apart)
            cut[1:] |= seen[1:] & seen[:-1]
    state = state.copy()
    t = 0
    while t < T:
        if not state[t]:
            t += 1
            continue
        e = t
        while e < T and state[e]:
            e += 1
        if (t > 0 and cut[t]) or (e < T and cut[e]):
            state[t:e] = 0
        t = e
    return state


def side_path(uv, conf, ks, side, H, b, height, s=SETTINGS):
    """Which of a pair of twin limbs is which in the panel looking across the creature, frame by frame, as one path
    through the clip: in every frame the cost of keeping the reader's labels and of swapping them is how far each
    keypoint then lands from one 3D point with the other panel (in twin_sigma of the height, capped at twin_cap per
    keypoint, so one wild reading cannot decide), and changing one's mind costs twin_cost. Frame 0 is the start image:
    its labels are right. Frames where any of the four readings is unsure say nothing. Returns (T,) 1 = swapped."""
    T, V = uv.shape[:2]
    sig, cap = float(s.get('twin_sigma', .02)) * height, float(s.get('twin_cap', 3.0))
    E = np.zeros((T, 2))
    for l, r in ks:
        ok = (conf[:, :, l] >= .3).all(1) & (conf[:, :, r] >= .3).all(1)
        pl = np.concatenate([uv[:, v, l] for v in range(V)], -1)                  # (T, 2V)
        pr = np.concatenate([uv[:, v, r] for v in range(V)], -1)
        ql, qr = pl.copy(), pr.copy()
        ql[:, 2 * side:2 * side + 2], qr[:, 2 * side:2 * side + 2] = pr[:, 2 * side:2 * side + 2], pl[:, 2 * side:2 * side + 2]
        for h, pts in enumerate(((pl, pr), (ql, qr))):
            for z in pts:
                E[:, h] += np.where(ok, np.minimum(np.linalg.norm((z - b) @ H.T, axis=1) / sig, cap), 0.0)
    S = float(s.get('twin_cost', 2.0))
    # a change of mind only where it puts no jump into the readings: where the reader itself jumped from one limb to the
    # other, or where the limbs cross. Two limbs the panel shows apart, each read smoothly from the frame before, swapped
    # there would both teleport (a tracker's turn: Seedance drew the right foot stepping in the front panel and the left
    # one in the side panel, frames 47-67; swapped, the planted foot jumped 86 px there and back)
    step = np.zeros(T)
    lim = float(s.get('twin_jump', 0.0))
    for t in range(1, T if lim > 0 else 1):
        d, n = 0.0, 0
        for l, r in ks:
            if min(conf[t - 1, side, l], conf[t - 1, side, r], conf[t, side, l], conf[t, side, r]) < .3:
                continue
            a0, b0, a1, b1 = uv[t - 1, side, l], uv[t - 1, side, r], uv[t, side, l], uv[t, side, r]
            d += (np.linalg.norm(b1 - a0) + np.linalg.norm(a1 - b0)) - (np.linalg.norm(a1 - a0) + np.linalg.norm(b1 - b0))
            n += 1
        if n and d > lim * n * sig:
            step[t] = np.inf
    acc = np.array([E[0, 0], np.inf])
    back = np.zeros((T, 2), int)
    for t in range(1, T):
        new = np.zeros(2)
        for st in (0, 1):
            stay, move = acc[st], acc[1 - st] + S + step[t]
            back[t, st] = st if stay <= move else 1 - st
            new[st] = min(stay, move) + E[t, st]
        acc = new
    state = np.zeros(T, int)
    state[-1] = int(np.argmin(acc))
    for t in range(T - 1, 0, -1):
        state[t - 1] = back[t, state[t]]
    return state


def bridge_unsure(uv, conf, s=SETTINGS):
    """A point the reader is unsure of (conf under bridge_conf: hidden, blurred, or read on its twin and counted down
    by fix_twins) for at most bridge_max frames, with sure readings of it on both sides: its readings there go along the
    straight line between those two, and keep their low weight. A hidden hand was guessed 100-300 px apart frame after
    frame and the fit, following the guesses a little each frame, shook the arm. Edits uv in place and returns it with
    the number of panel-points moved."""
    lim, longest = float(s.get('bridge_conf', .4)), int(s.get('bridge_max', 12))
    T, V, K = conf.shape
    moved = 0
    for v in range(V):
        for k in range(K):
            unsure = conf[:, v, k] < lim
            t = 0
            while t < T:
                if not unsure[t]:
                    t += 1
                    continue
                e = t
                while e < T and unsure[e]:
                    e += 1
                if t > 0 and e < T and e - t <= longest:
                    a, b = uv[t - 1, v, k].copy(), uv[e, v, k].copy()
                    for i in range(t, e):
                        f = (i - t + 1) / (e - t + 1)
                        uv[i, v, k] = a + (b - a) * f
                    moved += e - t
                t = e
    return uv, moved


def panels_agree(cr, clip, uv, conf, kps, s=SETTINGS, log=print):
    """After fix_twins: the panels' drift taken out (panel_drift), then the far limbs' guessed readings weighted down
    (panel_pick). Returns uv, conf."""
    if s.get('panel_drift', False):
        uv, _ = panel_drift(cr, clip, uv, conf, kps, log=log, least=s.get('drift_min', .05))
    if s.get('panel_pick', False):
        conf, _ = pick_panels(cr, clip, uv, conf, kps, s=s, log=log)
    return uv, conf


def shared_axis(views):
    """For two orthographic panels: the direction both see (across both view axes; armature space) and, per panel, the
    pixels a step of 1 along it moves a point. None for other cameras or panels looking the same way."""
    if len(views) != 2 or any(not hasattr(v['cam'], 'A') or v['cam'].type != 'ortho' for v in views):
        return None
    n = [np.cross(v['cam'].A[0], v['cam'].A[1]) for v in views]       # each panel's view axis
    e = np.cross(n[0], n[1])
    if np.linalg.norm(e) < .2 * np.linalg.norm(n[0]) * np.linalg.norm(n[1]):
        return None
    e /= np.linalg.norm(e)
    return e, [v['cam'].A @ e for v in views]


def panel_drift(cr, clip, uv, conf, kps, log=print, least=0.0):
    """uv with each panel's readings moved along the axis both panels see, half the centre line's offset between them
    each way per frame (median over the centre line's points both read surely, then a 5-frame median): the body
    agrees in both panels, where it was Seedance drew it. Not moved when the offset's median is under `least` of the
    height. And the offset per frame (in the first panel's pixels)."""
    views = cr.views(clip)
    sa = shared_axis(views)
    if sa is None:
        return uv, None
    _, g = sa
    b = [v['cam'].b for v in views]
    centre = [k for k, kp in enumerate(kps) if side_of(kp[0]) is None and not str(kp[1]).startswith('@')]
    T = len(uv)
    d = np.full(T, np.nan)
    for t in range(T):
        ok = [k for k in centre if conf[t, 0, k] >= .3 and conf[t, 1, k] >= .3]
        if len(ok) >= 3:
            s0 = (uv[t, 0, ok] - b[0]) @ g[0] / (g[0] @ g[0])
            s1 = (uv[t, 1, ok] - b[1]) @ g[1] / (g[1] @ g[1])
            d[t] = float(np.median(s0 - s1))
    if np.isnan(d).all():
        return uv, None
    known = np.nonzero(~np.isnan(d))[0]
    d = np.interp(np.arange(T), known, d[known])
    d = np.array([np.median(d[max(0, t - 2):t + 3]) for t in range(T)])
    px = np.abs(d) * np.linalg.norm(g[0])
    height_px = cr.height * float(np.linalg.norm(views[0]['cam'].A[0]))
    if np.median(px) < least * height_px:
        log(f'panels: the body along the axis both panels see is {np.median(px):.1f} px apart between them (median, '
            f'{100 * np.median(px) / height_px:.1f}% of the height): left as read')
        return uv, d * np.linalg.norm(g[0])
    uv = uv.copy()
    uv[:, 0] -= (d / 2)[:, None, None] * g[0]
    uv[:, 1] += (d / 2)[:, None, None] * g[1]
    log(f'panels: the body along the axis both panels see is {np.median(px):.1f} px apart between them (median, '
        f'{100 * np.median(px) / height_px:.1f}% of the height, {px.max():.1f} px at most); each panel moved half of it')
    return uv, d * np.linalg.norm(g[0])


def pick_panels(cr, clip, uv, conf, kps, s=SETTINGS, log=print):
    """conf with a limb point's reading weighted down in a panel that looks across the body and sees the limb behind
    its twin (the far side's limbs in a side panel: the reader guesses them, often where the near twin is), where the
    panels disagree on the point and another panel sees the limb in front of everything. Disagree: by more than
    pick_share of the height, and more than twice what the centre line's points disagree by in that frame; that
    frame, the one before and the one after. A limb behind something in every panel keeps its readings: nothing tells
    which panel guessed (a wolf's far hind knee is behind the chest in the front panel too; dropping the side panel's
    reading there cost the bench walk 0.5 deg). Nor does a panel that sees the twins apart lose any (a front panel's
    hind paw, behind the front legs: dropping it cost the bench gallop 1 deg on that leg). Returns conf and the mask
    (T, V, K)."""
    views = cr.views(clip)
    hit = np.zeros(conf.shape, bool)
    if len(views) < 2 or any(not hasattr(v['cam'], 'A') or v['cam'].type != 'ortho' for v in views):
        return conf, hit
    A = np.concatenate([v['cam'].A for v in views], 0)
    b = np.concatenate([v['cam'].b for v in views], 0)
    H = A @ np.linalg.pinv(A) - np.eye(len(A))
    labels = [k[0] for k in kps]
    rig = cr.rig
    chains = cr.limb_chains(as_list=True)
    if len(chains) < 2:
        return conf, hit
    H0 = rig.bone_points(cr.p0(None))[0]
    roots = H0[[ch[0] for ch in chains]]
    names = [rig.names[ch[0]] for ch in chains]
    lateral = np.cross(cr.up, cr.facing)
    behind = np.zeros((len(views), len(chains)), bool)            # (panel, limb): behind another limb's root
    guess = np.zeros((len(views), len(chains)), bool)             # (panel, limb): behind its twin, looking across
    for vi, v in enumerate(views):
        dep = v['depth'](roots)
        n = np.cross(v['cam'].A[0], v['cam'].A[1])
        across = abs(n @ lateral) > .7 * np.linalg.norm(n) * np.linalg.norm(lateral)
        for ci in range(len(chains)):
            if side_of(names[ci]) is None:                       # a neck or a tail: on the centre line, in sight
                continue
            for cj in range(len(chains)):
                if ci != cj and dep[ci] - dep[cj] > .5 * np.linalg.norm(roots[ci] - roots[cj]):
                    behind[vi, ci] = True
                    if across and names[cj] == twin_label(names[ci]):
                        guess[vi, ci] = True
    chain_of = {}
    for ci, ch in enumerate(chains):
        for j in ch:
            chain_of[j] = ci
    limb = np.full(len(kps), -1)
    for k, kp in enumerate(kps):
        if str(kp[1]).startswith('@'):
            continue
        j = rig.names.index(kp[1])
        while j >= 0 and j not in chain_of:                      # a toe or finger past the chain: its limb
            j = rig.parents[j]
        limb[k] = chain_of.get(j, -1)
        if limb[k] >= 0 and side_of(rig.names[chains[limb[k]][0]]) is None:
            limb[k] = -1                                         # a neck or a tail counts with the centre line
    centre = [k for k in range(len(kps)) if limb[k] < 0 and not str(kps[k][1]).startswith('@')]
    thr = s.get('pick_share', .12) * cr.height * float(np.mean([np.linalg.norm(v['cam'].A[0]) for v in views]))
    T, V = len(uv), len(views)

    def resid(t, k):
        z = np.concatenate([uv[t, o, k] for o in range(V)]) - b
        return float(np.linalg.norm(H @ z))
    for t in range(T):
        base = [resid(t, k) for k in centre if (conf[t, :, k] >= .3).all()]
        base = float(np.median(base)) if base else 0.0
        for k in np.nonzero(limb >= 0)[0]:
            c = limb[k]
            if not guess[:, c].any() or behind[:, c].all():
                continue
            front = [o for o in range(V) if not behind[o, c] and conf[t, o, k] >= .3]
            back = [o for o in range(V) if guess[o, c] and conf[t, o, k] >= s['conf_floor']]
            if not front or not back:
                continue
            r = resid(t, k)
            if r > thr and r > 2 * base:
                hit[t, back, k] = True
    none = np.zeros_like(hit[:1])
    hit = hit | np.r_[hit[1:], none] | np.r_[none, hit[:-1]]
    hit &= conf >= s['conf_floor']
    conf = conf.copy()
    conf[hit] *= s.get('pick_weight', .1)
    seen_from_behind = {views[vi]['name']: [names[ci] for ci in range(len(chains))
                                            if guess[vi, ci] and not behind[:, ci].all()] for vi in range(V)}
    if hit.any():
        per = {}
        for t, vi, k in zip(*np.nonzero(hit)):
            per[labels[k]] = per.get(labels[k], 0) + 1
        top = ', '.join(f'{lb} {c}' for lb, c in sorted(per.items(), key=lambda x: -x[1])[:8])
        log(f'panels: limbs seen from behind the body {seen_from_behind}; readings there the other panel contradicts, '
            f'weighted {s.get("pick_weight", .1)}: {int(hit.sum())} point-frames ({top})')
    else:
        log(f'panels: limbs seen from behind the body {seen_from_behind}; no reading there the other panel contradicts')
    return conf, hit


def prop_placer(cr, kps, kr, views):
    """For orthographic panels: placer(p, target, w) -> p with every prop moved where its read points are
    (triangulated from the panels that see them, the prop's own points then fitted rigidly: Kabsch; a log's points lie
    on a line, so only its axis turns, the shortest way), or None when no prop is read in two panels."""
    rig = cr.rig
    if not rig.props or any(not hasattr(v['cam'], 'A') or v['cam'].type != 'ortho' for v in views):
        return None
    names = [pr['name'] for pr in rig.props]
    sets = {}
    for k, kp in enumerate(kps):
        if str(kp[1]).startswith('@'):
            sets.setdefault(names.index(kp[1][1:]), []).append(k)
    lin = []
    for v in views:
        b = np.asarray(v['project'](np.zeros((1, 3))))[0]
        _, A = v['project'](np.zeros((1, 3)), True)
        lin.append((np.asarray(A)[0], b))

    def nearest_euler(R, ref):
        e = Rotation.from_matrix(R).as_euler('XYZ')
        best = None
        for c in (e, np.array([e[0] + np.pi, np.pi - e[1], e[2] + np.pi])):
            c = ref + (c - ref + np.pi) % (2 * np.pi) - np.pi
            if best is None or np.linalg.norm(c - ref) < np.linalg.norm(best - ref):
                best = c
        return best

    def place(p, target, w):
        q = p.copy()
        X_now = kr.points(p)
        moved = False
        for pk, ks in sets.items():
            pts, used = [], []
            for k in ks:
                rows, rhs = [], []
                for vi, (A, b) in enumerate(lin):
                    if w[vi, k] >= .3:
                        rows.append(A * w[vi, k])
                        rhs.append((target[vi, k] - b) * w[vi, k])
                if len(rows) >= 2:
                    pts.append(np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)[0])
                    used.append(k)
            if len(pts) == 1 and len(ks) == 1:                   # a round prop, read by its middle: moved, not turned
                base = rig.prop0 + 6 * pk
                q[base:base + 3] = p[base:base + 3] + pts[0] - X_now[used[0]]
                moved = True
                continue
            if len(pts) < 2:
                continue
            pts, now = np.array(pts), X_now[used]
            P0, P1 = now - now.mean(0), pts - pts.mean(0)
            sv = np.linalg.svd(P0, compute_uv=False)
            if len(pts) >= 3 and sv[1] > .05 * sv[0]:
                U, _, Vt = np.linalg.svd(P1.T @ P0)
                dR = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            else:                                                # points on a line: turn the axis only
                D = np.linalg.norm(P0[:, None] - P0[None], axis=2)
                i, j = np.unravel_index(np.argmax(D), D.shape)
                a0, a1 = P0[i] - P0[j], P1[i] - P1[j]
                a0, a1 = a0 / np.linalg.norm(a0), a1 / max(np.linalg.norm(a1), 1e-9)
                ax = np.cross(a0, a1)
                sn, cs = np.linalg.norm(ax), float(a0 @ a1)
                dR = np.eye(3) if sn < 1e-9 else Rotation.from_rotvec(ax / sn * np.arctan2(sn, cs)).as_matrix()
            base = rig.prop0 + 6 * pk
            R_new = dR @ rig.prop_rotation(p, pk)
            q[base + 3:base + 6] = nearest_euler(R_new, p[base + 3:base + 6])
            c0 = rig.prop_centre[pk]
            placed = (kr.rest[used] - c0) @ R_new.T + c0
            q[base:base + 3] = pts.mean(0) - placed.mean(0)
            moved = True
        return q if moved else None

    return place


def limb_reader(cr, kps, kr, views, one_panel=False):
    """For orthographic panels: reading_pose(p, target, w) -> p with every limb bone turned (the shortest way, from the
    hips or shoulders out) to point at its next joint where both panels read it, the body shifted onto the read root
    joint; the start the fit refines when the frame before got a limb wrong. one_panel: also where one panel reads
    it, the depth that panel cannot see from p (the frame before). None for other cameras."""
    rig = cr.rig
    if any(not hasattr(v['cam'], 'A') or v['cam'].type != 'ortho' for v in views):
        return None
    labels = [k[0] for k in kps]
    lin = []
    for v in views:
        b = np.asarray(v['project'](np.zeros((1, 3))))[0]
        _, A = v['project'](np.zeros((1, 3)), True)
        lin.append((np.asarray(A)[0], b))
    A_all = np.concatenate([A for A, _ in lin])
    b_all = np.concatenate([b for _, b in lin])
    Ap = np.linalg.pinv(A_all)
    H_res = A_all @ Ap - np.eye(len(A_all))
    plan = []                                            # (bone, its end keypoint, channel axes order, param indices)
    for chain in cr.limb_chains(as_list=True):
        for i, j in enumerate(chain):
            if j == rig.root or j not in rig.chan_of:
                continue
            chans = rig.chan_of[j]
            if sorted(a for a, _ in chans) != [0, 1, 2]:
                continue
            nxt = rig.names[chain[i + 1]] if i + 1 < len(chain) else rig.names[j] + '.tip'
            if nxt not in labels:
                continue
            plan.append((j, labels.index(nxt), ''.join('XYZ'[a] for a, _ in chans), [k for _, k in chans]))
    root_kp = labels.index(rig.names[rig.root]) if rig.names[rig.root] in labels else None

    def nearest(angles_new, ref):
        e = np.asarray(angles_new, float)
        best = None
        for c in (e, np.array([e[0] + np.pi, np.pi - e[1], e[2] + np.pi])):
            c = ref + (c - ref + np.pi) % (2 * np.pi) - np.pi
            if best is None or np.linalg.norm(c - ref) < np.linalg.norm(best - ref):
                best = c
        return best

    def reading_pose(p, target, w):
        good = (w >= .3).all(0)
        Y = np.full((len(labels), 3), np.nan)
        for k in np.nonzero(good)[0]:
            z = np.concatenate([target[vi, k] for vi in range(len(lin))]) - b_all
            if np.linalg.norm(H_res @ z) < 25.0:                 # both panels agree on the point
                Y[k] = Ap @ z
        if one_panel:
            X_before = kr.points(p)
            for k in np.nonzero((w >= .3).any(0) & ~good)[0]:
                rows = [r for vi in range(len(lin)) if w[vi, k] >= .3 for r in (2 * vi, 2 * vi + 1)]
                Ak = A_all[rows]
                z = np.concatenate([target[vi, k] for vi in range(len(lin))])[rows] - b_all[rows]
                lam = 1e-6 * float((Ak * Ak).sum())              # only the axis the panels do not see
                Y[k] = np.linalg.solve(Ak.T @ Ak + lam * np.eye(3), Ak.T @ z + lam * X_before[k])
        q = p.copy()
        if root_kp is not None and not np.isnan(Y[root_kp]).any():
            q[:3] += Y[root_kp] - kr.points(q)[root_kp]
        moved = 0
        for j, ke, seq, cols in plan:
            if np.isnan(Y[ke]).any():
                continue
            G, P, _ = rig.fk(q)
            head = P[j]
            cur = kr.points(q)[ke] - head
            want = Y[ke] - head
            if np.linalg.norm(cur) < 1e-6 or np.linalg.norm(want) < 1e-6:
                continue
            a0, a1 = cur / np.linalg.norm(cur), want / np.linalg.norm(want)
            ax = np.cross(a0, a1)
            sn, cs = np.linalg.norm(ax), float(a0 @ a1)
            if sn < 1e-6:
                continue
            dR = Rotation.from_rotvec(ax / sn * np.arctan2(sn, cs)).as_matrix()
            par = rig.parents[j]
            base = (G[par] if par >= 0 else np.eye(3)) @ rig.Rrl[j]
            local = base.T @ dR @ G[j]
            q[cols] = nearest(Rotation.from_matrix(local).as_euler(seq), q[cols])
            moved += 1
        return q if moved else None

    return reading_pose


def hidden_points(cr, clip, P, kps, uv=None, scale=.5):
    """(T, V, K) True where a keypoint's own limb does not show at its pixel in the fitted pose, through the clip's
    cameras: behind the body or another limb (what the reader learned to call hidden, measured on the fit). And, in a
    panel looking across the creature, a far-side limb's point read where the rest of the fitted body covers the
    picture (the trunk, the near limbs): the far limb cannot show there, the reading is a guess. The fitted pose alone
    did not catch it: an alien's far arm, read on its back, was fitted there, raised behind the back, where it shows."""
    from .visible import WINDOW, neighbours, triangle_bones
    rig = cr.rig
    kr = KeypointRig(cr, kps)
    tri_bone = triangle_bones(cr)
    allow = np.zeros((len(kps), rig.nb + len(rig.props)), bool)
    prop_names = [p['name'] for p in rig.props]
    for k, kp in enumerate(kps):
        if str(kp[1]).startswith('@'):
            allow[k, rig.nb + prop_names.index(kp[1][1:])] = True
        else:
            allow[k, list(neighbours(rig, rig.names.index(kp[1])))] = True
    tris = cr.mesh['tris']
    views = cr.views(clip)
    out = np.zeros((len(P), len(views), len(kps)), bool)
    # far-side limbs per panel looking across the body: (panel, triangles without the limb, its keypoints); which side
    # is far: the twin roots' depth in the start pose, their gap at least half the distance between them
    far = []
    H0 = rig.bone_points(cr.p0(None))[0]
    roots = {rig.names[ch[0]]: ch[0] for ch in cr.limb_chains(as_list=True)}
    for ch in cr.limb_chains(as_list=True):
        side = side_of(rig.names[ch[0]])
        if side is None:
            continue
        bones = set(ch)
        grow = True
        while grow:
            more = {k for k in range(rig.nb) if rig.parents[k] in bones} - bones
            bones |= more
            grow = bool(more)
        ks = [k for k, kp in enumerate(kps) if not str(kp[1]).startswith('@') and rig.names.index(kp[1]) in bones]
        keep = ~np.isin(tri_bone, sorted(bones))
        twin = roots.get(twin_label(rig.names[ch[0]]))
        if twin is None or not ks or uv is None:
            continue
        for v, view in enumerate(views):
            dm, dt = view['depth'](H0[[ch[0], twin]])
            if dm - dt > .5 * np.linalg.norm(H0[ch[0]] - H0[twin]):          # this limb lies behind its twin
                far.append((v, tris[keep], ks))
    for t, p in enumerate(P):
        X = rig.skin(p, cr.mesh['coords'], cr.idx, cr.wts)
        Xk = kr.points(p)
        for v, view in enumerate(views):
            w, h = view['cam'].size
            size = (max(1, int(w * scale)), max(1, int(h * scale)))
            tri, _, _ = cr.rasterize(view['project'](X) * scale, view['depth'](X), tris, size)
            bone = np.where(tri >= 0, tri_bone[np.maximum(tri, 0)], -1)
            pk = view['project'](Xk) * scale
            for k in range(len(kps)):
                x, y = int(pk[k, 0]), int(pk[k, 1])
                win = bone[max(0, y - WINDOW):y + WINDOW + 1, max(0, x - WINDOW):x + WINDOW + 1]
                win = win[win >= 0]
                out[t, v, k] = not (len(win) and allow[k, win].any())
        for v, tk, ks in far:
            view = views[v]
            w, h = view['cam'].size
            size = (max(1, int(w * scale)), max(1, int(h * scale)))
            tri, _, _ = cr.rasterize(view['project'](X) * scale, view['depth'](X), tk, size)
            for k in ks:
                x, y = int(uv[t, v, k, 0] * scale), int(uv[t, v, k, 1] * scale)
                if 0 <= x < size[0] and 0 <= y < size[1] and tri[y, x] >= 0:
                    out[t, v, k] = True
    return out


def body_picture(cr, clip, kps, scale=.5):
    """where(p, uv) -> (elsewhere, own), each (V, K): with the creature in pose p, its mesh rasterized through each
    panel's camera, what shows at the pixel where a keypoint is read (uv (V, K, 2)). own: its own limb segment (its
    bone, the bone above, the bones below: what the reader was taught to call seen, visible.py). elsewhere: the
    body, and none of its own segment: the reader points at another part of the body. A reading out on the
    background is neither (the joint may have moved there), and a prop is no part of the body (hands hold it).
    The cameras and the model are known, so what a panel can show of a joint is computed, not guessed."""
    from .visible import WINDOW, neighbours, triangle_bones
    rig = cr.rig
    tri_bone = triangle_bones(cr)
    allow = np.zeros((len(kps), rig.nb + len(rig.props)), bool)
    prop_names = [p['name'] for p in rig.props]
    for k, kp in enumerate(kps):
        if str(kp[1]).startswith('@'):
            allow[k, rig.nb + prop_names.index(kp[1][1:])] = True
        else:
            allow[k, list(neighbours(rig, rig.names.index(kp[1])))] = True
    tris = cr.mesh['tris']
    views = cr.views(clip)
    sizes = [(max(1, int(v['cam'].size[0] * scale)), max(1, int(v['cam'].size[1] * scale))) for v in views]

    def where(p, uv):
        X = rig.skin(p, cr.mesh['coords'], cr.idx, cr.wts)
        other = np.zeros((len(views), len(kps)), bool)
        own = np.zeros((len(views), len(kps)), bool)
        for v, view in enumerate(views):
            tri, _, _ = cr.rasterize(view['project'](X) * scale, view['depth'](X), tris, sizes[v])
            bone = np.where(tri >= 0, tri_bone[np.maximum(tri, 0)], -1)
            for k in range(len(kps)):
                if not np.isfinite(uv[v, k]).all():
                    continue
                x, y = int(uv[v, k, 0] * scale), int(uv[v, k, 1] * scale)
                if 0 <= x < sizes[v][0] and 0 <= y < sizes[v][1]:
                    win = bone[max(0, y - WINDOW):y + WINDOW + 1, max(0, x - WINDOW):x + WINDOW + 1]
                    own[v, k] = bool((win >= 0).any()) and bool(allow[k, win[win >= 0]].any())
                    win = win[(win >= 0) & (win < rig.nb)]
                    other[v, k] = bool(len(win)) and not allow[k, win].any()
        return other, own
    return where


def lowpass(x, hz, fps):
    """x (T, C) low-passed at hz, both ways (no lag)."""
    b, a = butter(2, min(hz / (.5 * fps), .99))
    return filtfilt(b, a, x, axis=0, padlen=min(8, len(x) - 2))


def smooth_turn(R, fps, s=SETTINGS):
    """The body's turn through a clip (the root's rotation vectors, (T, 3), as fitted frame by frame) cleaned up as a
    turn: unit quaternions made continuous in sign, a one-frame spike out and back taken out, low-passed like the
    bones (lowpass_hz, or lowpass_slow_hz for a body that never turns fast), back to rotation vectors kept continuous.
    A rotation vector filtered as three numbers jumps where the turn passes half a turn, and made continuous
    (unwrap_rotvecs) it jumps again where a spun body comes back near its start: there the vector's axis swings round
    with every small wobble (3 and 357 degrees in two frames), and the low-pass across the jump threw a body that the
    frame-by-frame fit had right onto its side or its back (the alien's pirouette, combo01, 101-190: up to 41 of the
    43 points both cameras plainly saw missed by 20-40% of the height; a weight changed by 0.0001 or a thread more
    flipped it, the frame-by-frame fit the same)."""
    R = np.asarray(R, float)
    T = len(R)
    if T < 3:
        return unwrap_rotvecs(R)
    q = Rotation.from_rotvec(R).as_quat()
    for t in range(1, T):
        if q[t] @ q[t - 1] < 0:
            q[t] = -q[t]
    # out and back in one frame by more than despike_deg: the quaternion's components bend by sin(angle / 2)
    q, _ = despike(q, np.full(4, 3), np.sin(np.radians(s['despike_deg']) / 2), 0.0)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    if s['lowpass_hz'] > 0 and T > 9:
        step = 2 * np.degrees(np.arccos(np.clip(np.abs((q[1:] * q[:-1]).sum(1)), 0, 1))) * fps     # deg/s
        hz = s['lowpass_hz']
        slow_hz = s.get('lowpass_slow_hz', 0)
        if slow_hz and slow_hz < hz and np.median(step) < s['slow_deg_s'] \
                and np.percentile(step, 90) < s.get('slow_p90_deg_s', np.inf):
            hz = slow_hz
            sm = lowpass(q, 3.0, fps)
            sm /= np.linalg.norm(sm, axis=1, keepdims=True)
            burst = 2 * np.degrees(np.arccos(np.clip(np.abs((sm[1:] * sm[:-1]).sum(1)), 0, 1))) * fps
            if s.get('burst_deg_s') and burst.max() > s['burst_deg_s']:
                hz = max(slow_hz, s.get('lowpass_burst_hz', slow_hz))
        q = lowpass(q, hz, fps)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
    return unwrap_rotvecs(Rotation.from_quat(q).as_rotvec())


def trunk_capsules(cr, s=SETTINGS):
    """[(bone, radius)] for every bone of the trunk (not in a limb chain, with skin of its own): radius self_core x the
    median distance of the vertices it moves most from its segment in the start pose, in space units."""
    rig = cr.rig
    in_chain = {j for ch in cr.limb_chains(as_list=True) for j in ch}
    p0 = cr.p0(None)
    H, T = rig.bone_points(p0)
    X = rig.skin(p0, cr.mesh['coords'], cr.idx, cr.wts)
    out = []
    for j in range(rig.nb):
        if j in in_chain:
            continue
        mine = (cr.idx[:, 0] == j) & (cr.wts[:, 0] > .5)
        if mine.sum() < 20:
            continue
        a, b = H[j], T[j]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        u = np.clip((X[mine] - a) @ ab / L2, 0, 1)
        d = np.linalg.norm(X[mine] - (a + u[:, None] * ab), axis=1)
        out.append((j, float(s.get('self_core', .7)) * float(np.median(d))))
    return out


class GuardRig:
    """Points that must stay out of the trunk, riding on their bones: every free limb's keypoints (not its first
    bone's), the middle of each of its bones, and on a hand holding something (its skin reaching over 2.5 times its
    bone's length: a sword) the object's middle and far end."""

    def __init__(self, cr, kps):
        rig = cr.rig
        self.rig = rig
        free = set()
        chains = []
        for ch in cr.limb_chains(as_list=True):
            if set(ch) & set(cr.legs) or any(re.search('wing', rig.names[j], re.I) or LOOSE.search(rig.names[j])
                                             for j in ch):
                continue
            free |= set(ch)
            chains.append(ch)
        rest, bone, names = [], [], []
        for k, kp in enumerate(kps):
            if str(kp[1]).startswith('@'):
                continue
            j = rig.names.index(kp[1])
            if j in free and not any(j == ch[0] for ch in chains):
                head = rig.rest_head[j]
                rest.append(np.asarray(kp[2], float) if is_offaxis(kp)
                            else head + float(kp[2]) * rig.rest_rot[j][:, 1] * rig.length[j])
                bone.append(j)
                names.append(kp[0])
        X = cr.mesh['coords']                                   # the rest pose, as the guard points ride on it
        for ch in chains:
            for j in ch[1:]:
                rest.append(rig.rest_head[j] + .5 * rig.rest_rot[j][:, 1] * rig.length[j])
                bone.append(j)
                names.append(rig.names[j] + '.mid')
            j = ch[-1]
            mine = (cr.idx[:, 0] == j) & (cr.wts[:, 0] > .5)
            if mine.sum() >= 20:
                d = np.linalg.norm(X[mine] - rig.rest_head[j], axis=1)
                if d.max() > 2.5 * max(float(rig.length[j]), 1e-6):
                    far = X[mine][int(np.argmax(d))]
                    rest.append(far)
                    bone.append(j)
                    names.append(rig.names[j] + '.held_tip')
                    rest.append(.5 * (far + rig.rest_head[j]))
                    bone.append(j)
                    names.append(rig.names[j] + '.held_mid')
        self.names = names
        self.rest = np.array(rest) if rest else np.zeros((0, 3))
        self.idx = np.zeros((len(rest), 4), np.int64)
        self.idx[:, 0] = bone
        self.wts = np.zeros((len(rest), 4))
        self.wts[:, 0] = 1.0

    def __len__(self):
        return len(self.rest)

    def points(self, p, jac=False):
        return self.rig.skin(p, self.rest, self.idx, self.wts, jac=jac)


def limb_points(cr, kps):
    """Keypoints on the free limbs (not a leg it stands on, not a wing): the points that must stay out of the trunk."""
    rig = cr.rig
    free = set()
    for ch in cr.limb_chains(as_list=True):
        if set(ch) & set(cr.legs) or any(re.search('wing', rig.names[j], re.I) or LOOSE.search(rig.names[j])
                                         for j in ch):
            continue
        free |= set(ch)
    return [k for k, kp in enumerate(kps) if not str(kp[1]).startswith('@') and rig.names.index(kp[1]) in free
            and rig.names.index(kp[1]) != next((ch[0] for ch in cr.limb_chains(as_list=True)
                                                 if rig.names.index(kp[1]) in ch), -1)]


def fit_hidden(cr, clip, uv, conf, kps, s=SETTINGS, log=print, narrow=None, dep=None):
    """fit_clip, then (hidden_refit) again with what the first fit's body hides in each panel weighted down: a far
    limb's depth comes from the other panel, the skeleton and its neighbours in time, not from the reader's guess."""
    P, rep = fit_clip(cr, clip, uv.copy(), conf.copy(), kps, s=s, log=log, narrow=narrow, dep=dep)
    if not s.get('hidden_refit', True) or len(cr.views(clip)) < 2:
        return P, rep
    hid = hidden_points(cr, clip, P, kps, uv=uv)
    conf = conf.copy()
    conf[hid] *= s.get('hidden_weight', .1)
    log(f'hidden: {[round(float(hid[:, v].mean()), 2) for v in range(hid.shape[1])]} of the points per panel; '
        f'fitted again with them weighted {s.get("hidden_weight", .1)}')
    P, rep = fit_clip(cr, clip, uv.copy(), conf, kps, s=s, log=log, narrow=narrow, dep=dep)
    if isinstance(rep, dict):
        rep['hidden_share'] = [round(float(hid[:, v].mean()), 3) for v in range(hid.shape[1])]
    return P, rep


def nearest_rotvec(r, ref):
    """Of the rotation vectors that turn the same as r (its angle +- 2 pi about its axis), the one nearest ref."""
    a = float(np.linalg.norm(r))
    if a < 1e-9:
        return r
    u = r / a
    return min((r, u * (a - 2 * np.pi), u * (a + 2 * np.pi), u * (a - 4 * np.pi), u * (a + 4 * np.pi)),
               key=lambda c: float(np.linalg.norm(c - ref)))


def unwrap_rotvecs(V):
    """A clip's rotation vectors (T, 3) made continuous: through half a turn a rotation vector jumps to the other
    side (180 and -180 degrees, the same turn); smoothed across the jump, an alien's spin turned back through 0
    halfway round."""
    V = np.array(V, float)
    for t in range(1, len(V)):
        V[t] = nearest_rotvec(V[t], V[t - 1])
    return V


def fit_clip(cr, clip, uv, conf, kps, s=SETTINGS, log=print, narrow=None, dep=None):
    """dep (T, V, K), or None: a depth reader's depth of every keypoint along each panel's line of sight, from the
    middle of the body's keypoints, in the creature's heights (NaN: not to be used); fitted as one more measurement
    (depth_w), its weight the reading's confidence, inside the same robust loss."""
    rig = cr.rig
    views = cr.views(clip)
    T = min(len(uv), len(cr.times(clip)))
    kr = KeypointRig(cr, kps)
    w_depth = float(s.get('depth_w', 0.0)) if dep is not None else 0.0
    if w_depth > 0:
        sights = []
        for v in views:                                   # each panel's line of sight (armature space) and its px per unit
            n = v['cam'].Ma.T @ v['cam'].fwd
            sights.append((n / np.linalg.norm(n), v['cam'].px_per_unit))
        body_k = np.array([not str(kp[1]).startswith('@') for kp in kps])
    lo, hi, p0 = cr.bounds(clip)
    lo, hi = lo.copy(), hi.copy()
    for c, (l, h) in (narrow or {}).items():
        lo[c], hi[c] = l, h
    prior_w = cr.prior(clip) * s['w_prior'] / 2
    smooth = np.full(rig.n, s['w_smooth'])
    smooth[:3] = s['w_smooth'] * 20                      # metres: 1 cm ~ 6 px-equivalent
    for k in range(len(rig.props)):                       # a prop's shift in metres, held lightly: a lifted log
        smooth[rig.prop0 + 6 * k:rig.prop0 + 6 * k + 3] = s['w_smooth'] * 2      # rises 3-8 px a frame
    P = np.zeros((T, rig.n))
    report = []
    prev = p0.copy()
    limb_bones = {j for ch in cr.limb_chains(as_list=True) for j in ch}
    limb_chans = [rig.index(rig.names[j], a) for j, a in rig.channels if j in limb_bones and j != rig.root]
    up, floor, w_floor = cr.up, cr.floor, s.get('w_floor', 0.0)
    pairs = end_pairs(kps)
    n_swapped = 0
    placer = prop_placer(cr, kps, kr, views)
    n_placed = 0
    reader_pose = limb_reader(cr, kps, kr, views, one_panel=s.get('read_one_panel', False)) \
        if s.get('read_start', True) else None
    n_read = 0
    root_rot = [rig.index(rig.names[rig.root], a) for a in (0, 1, 2) if (rig.root, a) in set(rig.channels)]
    n_turned = 0
    # the feet: each limb's end bone standing on the floor in the start pose, lying flat-ish (not a hoof pointing
    # down), with the axis across it that is level in the start pose, in the bone's own frame
    G0 = rig.fk(p0)[0]
    H0, T0 = rig.bone_points(p0)
    feet = []
    for ch in cr.limb_chains(as_list=True):
        j = ch[-1]
        d0 = T0[j] - H0[j]
        n0 = np.linalg.norm(d0)
        if n0 < 1e-9 or T0[j] @ up - floor > .05 * cr.height or abs(d0 @ up) / n0 > .8:
            continue
        lat0 = np.cross(up, d0 / n0)
        lat0 /= np.linalg.norm(lat0)
        chain_up = []
        k = j
        while k >= 0:
            chain_up.append(k)
            k = rig.parents[k]
        feet.append((j, G0[j].T @ lat0, set(chain_up)))
    w_flat = float(s.get('flat_feet', 0.0))
    # a reading the reader is sure of, moving smoothly through the frames around it (no jump onto another limb), is no
    # mistake: its miss keeps pulling up to stable_scale times further than cauchy_px. A crouching troll's far leg,
    # hidden in the side panel, drifted from its sure and still front-panel ankle by 43 px over 8 frames, the knee
    # straightened and the foot tipped to its limit, then snapped back (combo01, frames 8-16); every reading pulling
    # that far instead let wrong ones pull too (bench 2.99 -> 3.13 deg, injected swaps 67 -> 143 jumps)
    wide = float(s.get('stable_scale', 1.0)) if s['loss'] == 'cauchy' else 1.0
    if wide > 1:
        stable = plain_readings(cr, clip, uv, conf, s)

    def robust(C):
        """Cauchy with a scale per residual (least_squares' callable loss, f_scale 1)."""
        C2 = C ** 2

        def loss(z):
            q = z / C2
            return np.stack([C2 * np.log1p(q), 1 / (1 + q), -1 / (C2 * (1 + q) ** 2)])
        return loss

    def turned(q, deg):
        """q with the whole body turned deg about the up axis (the root's rotation vector composed with it)."""
        q = q.copy()
        rv = q[root_rot] if len(root_rot) == 3 else None
        if rv is None:
            return None
        q[root_rot] = nearest_rotvec((Rotation.from_rotvec(cr.up * np.radians(deg)) *
                                      Rotation.from_rotvec(rv)).as_rotvec(), rv)
        return q
    # who sees, decides (part model): what a panel can show of a joint is computed from the model, in the pose of
    # the frame before (body_picture). Where the two panels put a joint at different heights (the axis both see),
    # one of them is wrong: the one whose reading lies on the joint's own limb as the body stood a frame ago
    # decides, and a reading that lies on another part of the body is the reader's guess and is left out. A
    # centaur after its strike: its left arm, behind the body and the sword arm for the front panel, was "read"
    # there with conf .8 (the elbow, 50 px low) and .65 (the hand's tip, on the other shoulder's armour) while the
    # side panel saw the whole arm come down to the hilt; the fit followed the guess and the left hand flew off
    # the sword (attack01, 82-90). Left out, not counted a tenth: nothing else speaks of the axis that panel alone
    # sees, and a tenth still drew the hand across the panel. A guess nothing contradicts keeps its weight: a far
    # leg behind the body is the only word on how far forward it is.
    # "At different heights": further apart than the panels disagree anyway (twice the centre line's own
    # disagreement in the frame) and than counts as one place elsewhere here (twin_same_px of the height). At a
    # third of that a man lifting a log lost readings that were right: a hand moving before the chest is read a
    # little off where the frame before had it, on the chest's picture (bench 5.46 -> 5.87 deg).
    # Tried and dropped: a joint the fitted body hides from the panel, as the test (the hand, already dragged to
    # the wrong place, shows there; bench 39 -> 185 jumps)
    where = None
    ws_parts = set(str(s.get('who_sees_parts', '')).split(',')) if s.get('who_sees', False) else set()
    if 'model' in ws_parts and len(views) == 2 \
            and all(hasattr(v['cam'], 'A') and v['cam'].type == 'ortho' for v in views):
        where = body_picture(cr, clip, kps)
        A_all = np.concatenate([v['cam'].A for v in views], 0)
        b_all = np.concatenate([v['cam'].b for v in views], 0)
        H_res = A_all @ np.linalg.pinv(A_all) - np.eye(len(A_all))
        body_k = np.array([not str(kp[1]).startswith('@') for kp in kps])
        centre_k = np.array([side_of(kp[0]) is None for kp in kps]) & body_k
        # only the limbs' joints are judged: the trunk's readings place the whole body, and a few of them left out
        # turned a troll round for two stretches of 14 frames (throw01, 98-111 and 126-139)
        on_limb = {k for ch in cr.limb_chains(as_list=True) if side_of(rig.names[ch[0]]) is not None
                   for k in range(rig.nb) if k == ch[0] or rig.desc[ch[0], k]}
        limb_k = np.array([b and rig.names.index(kp[1]) in on_limb for kp, b in zip(kps, body_k)])
        same_px = float(s.get('twin_same_px', .06)) * cr.height * float(np.mean([np.linalg.norm(v['cam'].A[1])
                                                                                 for v in views]))
        guessed = np.zeros(len(kps), int)
    if pairs:
        uv, conf = uv.copy(), conf.copy()
    caps = trunk_capsules(cr, s) if s.get('no_self_pass', False) else []
    guard = GuardRig(cr, kps) if caps else None
    free_k = list(range(len(guard))) if guard is not None else []
    n_self = len(free_k) * len(caps)
    self_scale = float(s.get('self_w', 3000.0)) / cr.height
    if n_self:
        # a point already in (or at) a capsule in the start pose belongs there (a shoulder by the chest): exempt
        X0s = guard.points(p0)
        H0c, T0c = rig.bone_points(p0)
        self_on = np.ones(n_self)
        i = 0
        for k in free_k:
            for j, r in caps:
                a, b = H0c[j], T0c[j]
                ab = b - a
                u = min(max(float((X0s[k] - a) @ ab / max(float(ab @ ab), 1e-12)), 0.0), 1.0)
                if np.linalg.norm(X0s[k] - (a + u * ab)) < r + .02 * cr.height:
                    self_on[i] = 0.0
                i += 1
    n_pushed = 0
    for t in range(T):
        if pairs:
            # a prop's ends come read as "left, right" in each panel: each goes to the 3D end the frame before puts
            # nearer to it
            Xp = kr.points(prev)
            for v_i, v in enumerate(views):
                pp = v['project'](Xp)
                for k1, k2 in pairs:
                    keep = np.linalg.norm(uv[t, v_i, k1] - pp[k1]) + np.linalg.norm(uv[t, v_i, k2] - pp[k2])
                    swap = np.linalg.norm(uv[t, v_i, k1] - pp[k2]) + np.linalg.norm(uv[t, v_i, k2] - pp[k1])
                    if swap < keep:
                        uv[t, v_i, [k1, k2]] = uv[t, v_i, [k2, k1]]
                        conf[t, v_i, [k1, k2]] = conf[t, v_i, [k2, k1]]
                        n_swapped += 1
        w = np.clip(conf[t], 0, 1)
        w = np.where(w < s['conf_floor'], 0.0, w)       # (V, K)
        target = uv[t]
        if where is not None:
            other, own = where(prev, target)
            read = w >= .3
            off = np.linalg.norm((np.concatenate([target[v] for v in range(len(views))], -1) - b_all) @ H_res.T,
                                 axis=1)                # (K,) how far the two panels put each point apart
            ok = centre_k & read.all(0)
            base = float(np.median(off[ok])) if ok.sum() >= 3 else 0.0
            for v in range(len(views)):
                seen = np.any([own[o] & read[o] for o in range(len(views)) if o != v], axis=0)
                guess = other[v] & seen & limb_k & (w[v] > 0) & (off > max(2 * base, same_px))
                w[v, guess] = 0.0
                guessed += guess
        # feet on the floor the frame before lie flat in this one
        Tp = rig.bone_points(prev)[1]
        planted = [(j, loc, anc) for j, loc, anc in feet if w_flat > 0 and Tp[j] @ up - floor < .04 * cr.height]
        if w_depth > 0:                                    # the depth read: weighed like its point's reading
            d_t = dep[t] if t < len(dep) else np.full((len(views), len(kps)), np.nan)
            dw = np.where(np.isnan(d_t), 0.0, w) * w_depth
            if s.get('depth_where', 'unseen') == 'unseen' and len(views) == 2:
                dw = dw * (1 - np.clip(w[::-1], 0, 1))     # the other panel's reading of the joint, if sure, decides
            d_t = np.nan_to_num(d_t) * cr.height

        if n_self:
            Hc, Tc = rig.bone_points(prev)
            seg = [(Hc[j], Tc[j], r) for j, r in caps]

        def self_depth(X):
            """(n_self,) how far each free limb point is inside each trunk capsule (0 outside), and the outward unit
            directions (n_self, 3)."""
            dep_ = np.zeros(n_self)
            out_ = np.zeros((n_self, 3))
            i = 0
            for k in free_k:
                x = X[k]
                for a, b, r in seg:
                    ab = b - a
                    u = min(max(float((x - a) @ ab / max(float(ab @ ab), 1e-12)), 0.0), 1.0)
                    v = x - (a + u * ab)
                    dist = float(np.linalg.norm(v))
                    if dist < r:
                        dep_[i] = r - dist
                        out_[i] = v / dist if dist > 1e-9 else np.zeros(3)
                    i += 1
            return dep_, out_

        def resid(p):
            X = kr.points(p)
            r = [((v['project'](X) - target[k]) * w[k][:, None]).ravel() for k, v in enumerate(views)]
            r.append(smooth * (p - prev))
            r.append(prior_w * (p - p0))
            r.append(w_floor * np.minimum(X @ up - floor, 0.0))
            if planted:
                G = rig.fk(p)[0]
                r.append(np.array([w_flat * float((G[j] @ loc) @ up) for j, loc, _ in planted]))
            if w_depth > 0:
                mid = X[body_k].mean(0)
                for v_i, (n, kpx) in enumerate(sights):
                    r.append(dw[v_i] * kpx * ((X - mid) @ n - d_t[v_i]))
            if n_self:
                r.append(self_scale * self_on * self_depth(guard.points(p))[0])
            return np.concatenate(r)

        def jac(p):
            X, J = kr.points(p, jac=True)                # (K, 3), (K, 3, n)
            rows = []
            for k, v in enumerate(views):
                _, A = v['project'](X, True)            # (K, 2, 3)
                rows.append((np.einsum('kab,kbn->kan', A, J) * w[k][:, None, None]).reshape(-1, rig.n))
            rows.append(np.diag(smooth))
            rows.append(np.diag(prior_w))
            below = (X @ up - floor) < 0
            rows.append(w_floor * below[:, None] * np.einsum('d,kdn->kn', up, J))
            if planted:
                G, _, axes = rig.fk(p)
                F = np.zeros((len(planted), rig.n))
                for i, (j, loc, anc) in enumerate(planted):
                    v = G[j] @ loc
                    for jb, ax, k in axes:
                        if jb in anc:
                            F[i, k] = w_flat * float(np.cross(ax, v) @ up)
                rows.append(F)
            if w_depth > 0:
                Jd = J - J[body_k].mean(0)[None]
                for v_i, (n, kpx) in enumerate(sights):
                    rows.append((dw[v_i] * kpx)[:, None] * np.einsum('d,kdn->kn', n, Jd))
            if n_self:
                Gp, Jg = guard.points(p, jac=True)
                dep_, out_ = self_depth(Gp)
                Js = np.zeros((n_self, rig.n))
                i = 0
                for k in free_k:
                    for _ in caps:
                        if dep_[i] > 0 and self_on[i]:
                            Js[i] = -self_scale * out_[i] @ Jg[k]
                        i += 1
                rows.append(Js)
            return np.concatenate(rows)

        n_res = sum(len(kps) * 2 for _ in views)
        n_dep = len(views) * len(kps) if w_depth > 0 else 0
        lossf, fsc = s['loss'], s['cauchy_px']
        if wide > 1:
            C = np.full(n_res + 2 * rig.n + len(kps) + len(planted) + n_dep + n_self, float(s['cauchy_px']))
            if n_self:
                C[-n_self:] = 1e4                        # a limb inside the body pushes out in full
            C[:n_res] = np.repeat(np.where(stable[t], wide, 1.0).ravel(), 2) * float(s['cauchy_px'])
            lossf, fsc = robust(C), 1.0
        # two starts: the frame before, and the frame before with every limb back in its start pose (a limb the
        # frame before got wrong is not dragged along); the lower cost wins
        starts = [np.clip(prev if t else p0, lo, hi)]
        if t and s.get('two_starts', True):
            reset = prev.copy()
            reset[limb_chans] = p0[limb_chans]
            starts.append(np.clip(reset, lo, hi))
        read_i = place_i = None
        if reader_pose is not None and t:
            # also the limbs set where both panels read them: a limb the frame before got wrong (or that moved too
            # fast for the frame-to-frame fit) is picked up again instead of staying wrong for dozens of frames
            q = reader_pose(prev, target, w)
            if q is not None:
                if placer is not None:
                    placed = placer(q, target, w)
                    q = placed if placed is not None else q
                read_i = len(starts)
                starts.append(np.clip(q, lo, hi))
        if placer is not None and t:
            # also every prop set where both panels read it: the robust loss lets a prop that fell behind stay there
            # (a log lifted onto the shoulder stayed at the waist)
            q = placer(prev, target, w)
            if q is not None:
                place_i = len(starts)
                starts.append(np.clip(q, lo, hi))
        sols = [least_squares(resid, x0, jac=jac, bounds=(lo, hi), loss=lossf, f_scale=fsc,
                              max_nfev=60, x_scale='jac') for x0 in starts]
        best = int(np.argmin([r.cost for r in sols]))
        sol = sols[best]
        if t and s.get('turn_starts', True) and root_rot:
            r0 = sol.fun[:n_res].reshape(len(views), len(kps), 2)
            m0 = np.linalg.norm(r0, axis=-1) / np.maximum(w, 1e-6)
            sure = w > .3
            if sure.any() and float(np.median(m0[sure])) > s.get('turn_miss_px', 8.0):
                more = []
                for deg in (45, -45, 90, -90, 180):
                    q = turned(sol.x, deg)
                    if q is not None:
                        more.append(least_squares(resid, np.clip(q, lo, hi), jac=jac, bounds=(lo, hi), loss=lossf,
                                                  f_scale=fsc, max_nfev=60, x_scale='jac'))
                if more:
                    k = int(np.argmin([r.cost for r in more]))
                    if more[k].cost < sol.cost:
                        sols.append(more[k])
                        best = len(sols) - 1
                        sol = more[k]
                        n_turned += 1
        n_placed += int(best == place_i)
        n_read += int(best == read_i)
        P[t] = sol.x
        if n_self and (self_on * self_depth(guard.points(sol.x))[0]).max() > 0:
            n_pushed += 1
        prev = sol.x
        r = sol.fun[:n_res].reshape(len(views), len(kps), 2)
        miss = np.linalg.norm(r, axis=-1) / np.maximum(w, 1e-6)
        seen = w > .3
        report.append(dict(frame=t, start=['prev', 'reset', 'read', 'prop'][[0, 1, read_i, place_i].index(best)] if best in (0, 1, read_i, place_i) else best, median_miss_px=round(float(np.median(miss[seen])), 2) if seen.any() else None,
                           unsure_both=int(((w < .3).all(0)).sum())))
    if n_self:
        log(f'fit {clip}: free limbs kept out of {len(caps)} trunk capsules ({len(free_k)} points); still touching '
            f'inside after the solve in {n_pushed} frames')
    x, changed, fps = clean_motion(cr, clip, P, uv, conf, kps, lo, hi, s=s, log=log)
    if pairs:
        log(f'fit {clip}: prop ends assigned by the frame before, swapped in {n_swapped} panel-frames')
    if n_turned:
        log(f'fit {clip}: the body turned about the up axis won in {n_turned} badly fitted frames')
    if placer is not None:
        log(f'fit {clip}: the prop placed from its readings won in {n_placed} frames')
    if reader_pose is not None:
        log(f'fit {clip}: the limbs set from the readings won in {n_read} frames')
    if where is not None:
        top = ', '.join(f'{kps[k][0]} {guessed[k]}' for k in np.argsort(-guessed)[:8] if guessed[k])
        log(f'fit {clip}: readings that lie on another part of the body, at another height than the panel that '
            f'reads the joint on its own limb, left out: {int(guessed.sum())} ({top})')
    log(f'fit {clip}: {T} frames; median keypoint miss {np.median([r["median_miss_px"] or 0 for r in report]):.2f} px; '
        f'{changed} one-frame spikes smoothed; low-pass {s["lowpass_hz"]} Hz at {fps:.0f} fps')
    return x, report


def clean_motion(cr, clip, P, uv, conf, kps, lo, hi, s=SETTINGS, log=print):
    """The mocap clean-up of a frame-by-frame fit P (T, n): the body's turn made continuous and calmed as a turn,
    one-frame spikes taken out (unless the reading backs them), the low-pass, a limb end under the floor bent out
    by its own limb, the body lifted out of the floor along one smooth curve. uv, conf: the readings as the fit used
    them. Returns (the motion, the number of one-frame spikes smoothed, the clip's frame rate)."""
    rig = cr.rig
    views = cr.views(clip)
    T = len(P)
    kr = KeypointRig(cr, kps)
    root_rot = [rig.index(rig.names[rig.root], a) for a in (0, 1, 2) if (rig.root, a) in set(rig.channels)]
    up, floor, w_floor = cr.up, cr.floor, s.get('w_floor', 0.0)
    # mocap clean-up, as the tool's solve (the body's turn made continuous first)
    x = P.copy()
    times = cr.times(clip)[:T]
    fps = 1.0 / max(np.median(np.diff(times)), 1e-6) if T > 1 else 24.0
    turn = smooth_turn(P[:, root_rot], fps, s) if len(root_rot) == 3 and s.get('turn_quat', False) else None
    if turn is not None:
        x[:, root_rot] = turn
    elif len(root_rot) == 3:
        x[:, root_rot] = unwrap_rotvecs(x[:, root_rot])
    x, changed = despike(x, np.arange(rig.n), np.radians(s['despike_deg']), .03 * cr.height)
    if turn is not None:
        x[:, root_rot] = turn
    kept = 0
    if s.get('despike_keep_px', 0) > 0:
        def miss(q, t, ks=None, sure=.5):
            X = kr.points(q)
            m = np.stack([np.linalg.norm(v['project'](X) - uv[t, k], axis=1) for k, v in enumerate(views)])
            use = conf[t] > sure
            if ks is not None:
                use &= np.isin(np.arange(len(kps)), ks)[None]
            m = m[use]
            return float(np.median(m)) if len(m) else 0.0
        # per limb too: a limb's fast swing the reading backs is not a spike either, though the median over the whole
        # body barely moves (the boss's right arm flapping back along its flank for 1-2 frames in every 6)
        limbs = []
        if s.get('despike_limbs', True):
            for ch in cr.limb_chains(as_list=True):
                bones = set(ch)
                grow = True
                while grow:
                    more = {k for k in range(rig.nb) if rig.parents[k] in bones} - bones
                    bones |= more
                    grow = bool(more)
                cols = sorted({c for j in bones for _, c in rig.chan_of.get(j, [])})
                ks = [k for k, kp in enumerate(kps) if not str(kp[1]).startswith('@') and rig.names.index(kp[1]) in bones]
                if cols and ks:
                    limbs.append((cols, ks))
        kept_limb = 0
        for t in range(T):
            if not np.any(np.abs(x[t] - P[t]) > 1e-9):
                continue
            if miss(x[t], t) - miss(P[t], t) > s['despike_keep_px']:
                x[t] = P[t]
                kept += 1
                continue
            for cols, ks in limbs:
                if np.any(np.abs(x[t, cols] - P[t, cols]) > 1e-9):
                    q = x[t].copy()
                    q[cols] = P[t, cols]
                    if miss(x[t], t, ks, .3) - miss(q, t, ks, .3) > s['despike_keep_px']:
                        x[t] = q
                        kept_limb += 1
        if turn is not None:
            x[:, root_rot] = turn
        changed_frames = int(np.any(np.abs(x - P) > 1e-9, axis=1).sum())
        log(f'fit {clip}: one-frame spikes smoothed in {changed_frames} frames; {kept} more the reading backs, kept, '
            f'and {kept_limb} limbs in others')
    if s['lowpass_hz'] > 0 and T > 9:
        fast = lowpass(x, s['lowpass_hz'], fps)
        slow_hz = s.get('lowpass_slow_hz', 0)
        if slow_hz and slow_hz < s['lowpass_hz']:
            slow = lowpass(x, slow_hz, fps)
            burst_hz = max(slow_hz, s.get('lowpass_burst_hz', slow_hz))
            burst = lowpass(x, burst_hz, fps) if s.get('burst_deg_s') and burst_hz > slow_hz else None
            path = lowpass(x, 3.0, fps) if burst is not None else None
            n_slow = n_burst = 0
            for j, chans in rig.chan_of.items():
                cols = [k for _, k in chans]
                sp = np.degrees(np.linalg.norm(np.diff(x[:, cols], axis=0), axis=1)) * fps
                if np.median(sp) < s['slow_deg_s'] and np.percentile(sp, 90) < s.get('slow_p90_deg_s', np.inf):
                    peak = np.degrees(np.linalg.norm(np.diff(path[:, cols], axis=0), axis=1)).max() * fps \
                        if burst is not None else 0.0
                    if peak > s.get('burst_deg_s', np.inf):
                        fast[:, cols] = burst[:, cols]
                        n_burst += 1
                    else:
                        fast[:, cols] = slow[:, cols]
                        n_slow += 1
            log(f'fit {clip}: {n_slow} of {len(rig.chan_of)} bones slower than {s["slow_deg_s"]:.0f} deg/s (and never fast): '
                f'low-pass {slow_hz} Hz' + (f'; {n_burst} more slow but with a burst: {burst_hz} Hz' if n_burst else ''))
        x = fast
    x = np.clip(x, lo, hi)
    if turn is not None:
        x[:, root_rot] = turn
    body = [k for k, kp in enumerate(kps) if not str(kp[1]).startswith('@')]     # a prop is no reason to lift him
    # a limb end under the floor comes out by bending its own limb, the body staying where it is; the body lifts only
    # for what is left. A wolf's curling front paw (its toes now a bone along the sole) dipped 5 cm under the floor
    # and the whole wolf was lifted 3-4 cm for 14 frames, every standing paw in the air (walk01, frames 36-49). The
    # fit's own floor term is inside the robust loss, so a deep dip barely pulls there; here it pulls in full.
    n_limb = 0
    if w_floor > 0:
        bone_of = [rig.names.index(kp[1]) if not str(kp[1]).startswith('@') else -1 for kp in kps]
        chains = cr.limb_chains(as_list=True)
        for t in range(T):
            X0 = kr.points(x[t])
            under = [k for k in body if X0[k] @ up < floor - .002 * cr.height]
            for ch in chains:
                if not any(bone_of[k] in ch for k in under):
                    continue
                cols = sorted({c for j in ch for _, c in rig.chan_of.get(j, [])})
                if not cols:
                    continue
                q = x[t].copy()
                wt = np.clip(conf[t], 0, 1)
                wt = np.where(wt < s['conf_floor'], 0.0, wt)
                if s.get('floor_bend', 'still') == 'read':
                    ref = [uv[t, k] for k in range(len(views))]
                    r0 = np.stack([np.linalg.norm(v['project'](X0) - ref[k], axis=1) for k, v in enumerate(views)])
                    wt = wt / np.sqrt(1 + (r0 / s['cauchy_px']) ** 2)      # far misses barely pull, as in the fit
                else:
                    ref = [v['project'](X0) for v in views]                 # the low-passed pose, as the cameras see it

                def out_of_floor(z, q=q, cols=cols, wt=wt, ref=ref):
                    p = q.copy()
                    p[cols] = z
                    Xp = kr.points(p)
                    r = [((v['project'](Xp) - ref[k]) * wt[k][:, None]).ravel() for k, v in enumerate(views)]
                    r.append(10 * w_floor * np.minimum(Xp[body] @ up - floor, 0.0))
                    r.append(s['w_smooth'] * (z - q[cols]))
                    return np.concatenate(r)
                x[t, cols] = least_squares(out_of_floor, q[cols], bounds=(lo[cols], hi[cols]), max_nfev=40).x
                X0 = kr.points(x[t])
                n_limb += 1
        if n_limb:
            log(f'fit {clip}: a limb end under the floor bent out of it by its own limb in {n_limb} limb-frames')
    # what is left: the body lifted out of the floor along ONE smooth curve, never less than a frame needs (lifting
    # each frame on its own made the wolf hop 3-6 cm up and down a frame: run01, frames 26-29)
    dip = np.array([floor - float((kr.points(x[t])[body] @ up).min()) for t in range(T)])
    need = np.maximum(dip, 0.0)
    lift = need.copy()
    if w_floor > 0 and need.any():
        k = np.exp(-.5 * (np.arange(-6, 7) / 2.0) ** 2)
        k /= k.sum()
        for _ in range(60):                                   # smooth, then never below the need
            lift = np.maximum(np.convolve(np.pad(lift, 6, mode='edge'), k, mode='valid'), need)
        x[:, :3] += lift[:, None] * up[None]
    log(f'fit {clip}: lifted out of the floor in {int((need > 0).sum())} frames, at most {100 * need.max():.1f} cm, '
        f'along a smooth curve (largest change a frame {100 * float(np.abs(np.diff(lift)).max()) if T > 1 else 0:.1f} cm)')
    travel = 0.0
    if s.get('steps', False):
        x, travel = root_from_steps(cr, clip, x, fps, s=s, log=log)
    if s.get('foot_lock', False):
        x = lock_feet(cr, clip, x, uv, conf, kps, lo, hi, fps, s={**s, 'lock_video': travel == 0.0}, log=log)
    if s.get('springs', False):
        x = spring_pass(cr, clip, x, fps, lo, hi, s=s, log=log)
    if s.get('no_self_pass', False):
        x = keep_out(cr, clip, x, kps, lo, hi, s=s, log=log)
    return x, changed, fps


def keep_out(cr, clip, x, kps, lo, hi, s=SETTINGS, log=print):
    """Every frame of the finished motion: a guard point (GuardRig) inside a trunk capsule (and not inside it in the
    start pose) comes out by the least change of its own limb's joints."""
    caps = trunk_capsules(cr, s)
    guard = GuardRig(cr, kps)
    if not caps or not len(guard):
        return x
    rig = cr.rig
    p0 = cr.p0(None)
    G0 = guard.points(p0)
    H0, T0 = rig.bone_points(p0)

    def depths(G, H, T):
        out = np.zeros((len(G), len(caps)))
        for c, (j, r) in enumerate(caps):
            a, b = H[j], T[j]
            ab = b - a
            u = np.clip((G - a) @ ab / max(float(ab @ ab), 1e-12), 0, 1)
            out[:, c] = r - np.linalg.norm(G - (a + u[:, None] * ab), axis=1)
        return out
    on = depths(G0, H0, T0) < -.02 * cr.height                         # pairs that are outside at rest
    chain_of = {}
    for ch in cr.limb_chains(as_list=True):
        for j in ch:
            chain_of[j] = ch
    scale = float(s.get('self_w', 3000.0)) / cr.height
    out = x.copy()
    n_fixed = 0
    for t in range(len(x)):
        H, T = rig.bone_points(out[t])
        d = np.where(on, depths(guard.points(out[t]), H, T), -1.0)
        if d.max() <= .005 * cr.height:
            continue
        bones = {guard.idx[g, 0] for g in np.nonzero((d > 0).any(1))[0]}
        cols = sorted({c for b in bones for jj in chain_of.get(b, [b]) for _, c in rig.chan_of.get(jj, [])})
        if not cols:
            continue
        q = out[t].copy()

        def resid(z, q=q, cols=cols, H=H, T=T):
            pp = q.copy()
            pp[cols] = z
            dd = np.where(on, depths(guard.points(pp), H, T), -1.0)
            return np.concatenate([scale * np.maximum(dd, 0).ravel(), s['w_smooth'] * (z - q[cols])])
        out[t, cols] = least_squares(resid, q[cols], bounds=(lo[cols], hi[cols]), max_nfev=40).x
        n_fixed += 1
    if n_fixed:
        log(f'fit {clip}: a limb or what it holds pushed out of the body after the clean-up in {n_fixed} frames')
    return out


def root_from_steps(cr, clip, x, fps, s=SETTINGS, log=print):
    """The body moved by its steps: Seedance often draws a walk or a run on the spot (the body stays, the standing feet
    slide back, in the video too). Where the low feet (within steps_low of the height above where they rest) slide
    back against the body's facing (faster than steps_back heights a second), the body is moved forward along the
    floor by what they slide, smoothed at steps_hz; between steps (a run's flight) the pace goes on. Only in a clip
    that walks on the spot: low feet sliding back in steps_share of the frames, steps_same_way of their slide one way
    (a fight's feet shuffle every way: 0.70-0.88 on the laptop's clips, walks and runs 0.93-1.00). Returns the motion
    and how far the body went (heights; 0 when left as is)."""
    rig = cr.rig
    T = len(x)
    if T < 10:
        return x, 0.0
    up, floor, height = cr.up, cr.floor, cr.height
    p0 = cr.bounds(clip)[2]
    H0, T0 = rig.bone_points(p0)
    chains = [ch for ch in cr.limb_chains(as_list=True) if set(ch) <= set(cr.legs)]
    root_rot = [rig.index(rig.names[rig.root], a) for a in (0, 1, 2) if (rig.root, a) in set(rig.channels)]
    if not chains:
        return x, 0.0
    HT = np.array([np.stack(rig.bone_points(p)) for p in x])
    fac = np.array([Rotation.from_rotvec(p[root_rot]).apply(cr.facing) if len(root_rot) == 3 else cr.facing
                    for p in x])
    flat = lambda v: v - (v @ up)[..., None] * up
    fac = flat(fac)
    fac /= np.maximum(np.linalg.norm(fac, axis=1, keepdims=True), 1e-9)
    vroot = np.gradient(lowpass(flat(HT[:, 0, rig.root]), 3.0, fps), axis=0) * fps
    slides, weights, every = np.zeros((T, 3)), np.zeros(T), []
    for ch in chains:
        j = ch[-1]
        rest = np.array([H0[j] @ up - floor, T0[j] @ up - floor])
        ends = HT[:, :, j]
        down = ((ends @ up - floor) - rest[None] < s.get('steps_low', .04) * height).all(1)
        v = np.gradient(lowpass(flat(ends.mean(1)), 3.0, fps), axis=0) * fps
        stance = down & (((v - vroot) * fac).sum(1) < -s.get('steps_back', .08) * height)
        slides[stance] += v[stance]
        weights[stance] += 1
        every.append(v[stance])
    have = weights > 0
    V = np.concatenate(every)
    way = float(np.linalg.norm(V.mean(0)) / max(np.mean(np.linalg.norm(V, axis=1)), 1e-9)) if len(V) else 0.0
    pace = np.linalg.norm(slides[have] / weights[have, None], axis=1) if have.any() else np.zeros(1)
    spread = float(np.std(pace) / max(np.mean(pace), 1e-9))
    if (have.mean() < s.get('steps_share', .55) or way < s.get('steps_same_way', .92)
            or spread > s.get('steps_steady', .5)):
        log(f'fit {clip}: low feet slide back in {100 * have.mean():.0f}% of the frames, {way:.2f} of it one way, '
            f'pace spread {spread:.2f}: no walk on the spot')
        return x, 0.0
    corr = np.zeros((T, 3))
    corr[have] = -slides[have] / weights[have, None]
    for d in range(3):
        corr[~have, d] = np.interp(np.nonzero(~have)[0], np.nonzero(have)[0], corr[have, d])
    offset = flat(np.cumsum(lowpass(corr, s.get('steps_hz', 2.0), fps), axis=0) / fps)
    offset -= offset[0]
    out = x.copy()
    out[:, :3] += offset
    travel = float(np.linalg.norm(offset[-1])) / height
    log(f'fit {clip}: a walk on the spot ({100 * have.mean():.0f}% of the frames, {way:.2f} one way): the body moved '
        f'{travel:.2f} heights by its steps')
    return out, travel


LOOSE = re.compile(r'(tail|(^|[^a-z])ear([._0-9]|$))', re.I)


def spring_pass(cr, clip, x, fps, lo, hi, s=SETTINGS, log=print):
    """Springs on the loose parts (a tail, an ear, by their names): each loose chain's joints follow where the motion
    puts them through a damped spring (spring_hz, spring_zeta; the tip spring_soft softer), simulated between the
    frames with the bones' lengths kept, then the bones turned onto the simulated joints. The part keeps what the
    video gave it, but nothing on it jolts faster than the spring moves: the reading's shake becomes follow-through.
    Not wings: a wing's flap is the motion itself, and springs on its outer bones broke it."""
    rig = cr.rig
    T = len(x)
    if T < 5:
        return x
    height = cr.height
    hz, zeta, soft, sub = s.get('spring_hz', 4.0), s.get('spring_zeta', .55), s.get('spring_soft', .4), 4
    out = x.copy()
    n_chains = 0
    for ch in cr.limb_chains(as_list=True):
        first = next((j for j in ch if LOOSE.search(rig.names[j]) and j in rig.chan_of), None)
        if first is None:
            continue
        loose = [j for j in ch[ch.index(first):] if j in rig.chan_of]
        cols = sorted({c for j in loose for _, c in rig.chan_of[j]})
        HT = np.array([np.stack(rig.bone_points(p)) for p in out])
        anchor, goal = HT[:, 0, loose[0]], HT[:, 1, loose]
        n = len(loose)
        length = np.linalg.norm(goal[0] - np.vstack([anchor[0], goal[0, :-1]]), axis=1)
        w = 2 * np.pi * hz * (1 - soft * np.arange(n) / max(n - 1, 1))
        P, V = goal[0].copy(), np.zeros((n, 3))
        sim = np.zeros_like(goal)
        sim[0] = P
        dt = 1.0 / (fps * sub)
        for t in range(1, T):
            for k in range(1, sub + 1):
                f = k / sub
                g = goal[t - 1] * (1 - f) + goal[t] * f
                prev = anchor[t - 1] * (1 - f) + anchor[t] * f
                V = V + ((w ** 2)[:, None] * (g - P) - (2 * zeta * w)[:, None] * V) * dt
                Pn = P + V * dt
                for i in range(n):                                   # the bones keep their lengths, root outward
                    d = Pn[i] - prev
                    Pn[i] = prev + d / max(np.linalg.norm(d), 1e-9) * length[i]
                    prev = Pn[i]
                V, P = (Pn - P) / dt, Pn
            sim[t] = P
        scale = 2000.0 / height
        for t in range(1, T):
            q = out[t].copy()

            def resid(z, q=q, t=t):
                p = q.copy()
                p[cols] = z
                return np.concatenate([scale * (rig.bone_points(p)[1][loose] - sim[t]).ravel(), 5.0 * (z - q[cols])])
            out[t, cols] = least_squares(resid, q[cols], bounds=(lo[cols], hi[cols]), max_nfev=30).x
        n_chains += 1
    if n_chains:
        log(f'fit {clip}: springs on {n_chains} loose chains (tails, ears), {hz} Hz')
    return out


def path_speed(X, fps, hz=3.0):
    """(T, ...) speed of points X (T, ..., D) along their own hz path, in X's units a second."""
    T = len(X)
    if T < 10:
        return np.zeros(X.shape[:-1])
    S = lowpass(X.reshape(T, -1), hz, fps).reshape(X.shape)
    v = np.zeros(X.shape[:-1])
    v[1:-1] = np.linalg.norm(S[2:] - S[:-2], axis=-1) * fps / 2
    v[0], v[-1] = v[1], v[-2]
    return v


def lock_feet(cr, clip, x, uv, conf, kps, lo, hi, fps, s=SETTINGS, log=print):
    """A foot that stands in the video stands in the motion: his first rule, nothing slides that no leg moves. Per
    leg the creature stands on (cr.legs), its last bone (toes along the sole, a hoof) is down in a frame when both its
    ends are within lock_height of the height above where they rest in the start pose, and still when it moves slower
    than lock_speed heights a second in the motion and, where a panel reads its keypoints surely, in the video too.
    Over every stretch of at least lock_frames such frames the bone is pinned where it stands (the stretch's median,
    on the floor at its resting height) by the least change of its own leg's joints, easing in and out over
    lock_blend frames. The heel may lift: only the last bone is pinned."""
    from scipy.optimize import least_squares as lsq
    rig = cr.rig
    T = len(x)
    if T < 5:
        return x
    views = cr.views(clip)
    up, floor, height = cr.up, cr.floor, cr.height
    p0 = cr.bounds(clip)[2]
    H0, T0 = rig.bone_points(p0)
    chains = [ch for ch in cr.limb_chains(as_list=True) if set(ch) <= set(cr.legs)]
    kp_of = {}
    for k, kp in enumerate(kps):
        if not str(kp[1]).startswith('@'):
            kp_of.setdefault(rig.names.index(kp[1]), []).append(k)
    hp = [v['cam'].px_per_unit * height for v in views]
    HT = np.array([np.stack(rig.bone_points(p)) for p in x])          # (T, 2, bones, 3) heads, tails
    lh, ls = float(s.get('lock_height', .03)), float(s.get('lock_speed', .3))
    least, ease = int(s.get('lock_frames', 3)), int(s.get('lock_blend', 3))
    out = x.copy()
    n_runs = n_frames = 0
    worst = 0.0
    for ch in chains:
        j = ch[-1]
        cols = sorted({c for b in ch for _, c in rig.chan_of.get(b, [])})
        if not cols:
            continue
        rest = np.array([H0[j] @ up - floor, T0[j] @ up - floor])
        ends = HT[:, :, j]                                               # (T, 2, 3)
        h = ends @ up - floor
        down = (h - rest[None] < lh * height).all(1)
        still = path_speed(ends.mean(1), fps) / height < ls
        ks = [k for b in ch[-2:] for k in kp_of.get(b, [])]               # the video's word: the foot's own keypoints
        for v_i, v in enumerate(views):
            if hp[v_i] is None or not ks or not s.get('lock_video', True):
                continue
            sure = (conf[:T, v_i][:, ks] >= .5).all(1)
            vv = path_speed(uv[:T, v_i][:, ks], fps).max(1) / hp[v_i]
            still &= ~sure | (vv < ls)
        on = down & still
        on[1:-1] |= on[:-2] & on[2:]                                     # a one-frame gap closes
        runs, t = [], 0
        while t < T:
            if on[t]:
                e = t
                while e + 1 < T and on[e + 1]:
                    e += 1
                if e - t + 1 >= least:
                    runs.append((t, e))
                t = e + 1
            else:
                t += 1
        if not runs:
            continue
        w = np.zeros(T)
        target = np.zeros((T, 2, 3))
        for a, b in runs:
            tgt = np.median(ends[a:b + 1], axis=0)
            tgt += (rest - tgt @ up)[:, None] * up[None]                # on the floor, at its resting height
            for t in range(max(0, a - ease), min(T, b + ease + 1)):
                wt = 1.0 if a <= t <= b else 1 - (a - t if t < a else t - b) / (ease + 1)
                if wt > w[t]:
                    w[t], target[t] = wt, tgt
            n_runs += 1
        scale = 2000.0 / height
        for t in np.nonzero(w > 0)[0]:
            goal = ends[t] + w[t] * (target[t] - ends[t])
            q = out[t].copy()

            def resid(z, q=q, goal=goal):
                p = q.copy()
                p[cols] = z
                Hb, Tb = rig.bone_points(p)
                return np.concatenate([scale * (np.stack([Hb[j], Tb[j]]) - goal).ravel(),
                                       s['w_smooth'] * (z - q[cols])])
            z = lsq(resid, q[cols], bounds=(lo[cols], hi[cols]), max_nfev=40).x
            out[t, cols] = z
            Hb, Tb = rig.bone_points(out[t])
            worst = max(worst, float(np.linalg.norm(np.stack([Hb[j], Tb[j]]) - goal, axis=1).max()) / height)
            n_frames += 1
    log(f'fit {clip}: feet pinned where they stand in {n_runs} stretches, {n_frames} leg-frames; farthest a pinned foot '
        f'stayed from its place {100 * worst:.1f}% of the height')
    return out


def main():
    ap = argparse.ArgumentParser(prog='video2rig.reader.fit')
    ap.add_argument('project')
    ap.add_argument('clip')
    ap.add_argument('--reading', default='', help='work/<clip>/reader.npz (default)')
    ap.add_argument('--tag', default='-reader')
    ap.add_argument('--no-twins', action='store_true', help='keep the readings as read (no left/right check, no '
                    'twin jumps, no bridging)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='one SETTINGS value for this run')
    a = ap.parse_args()
    from .road import fit_reading
    settings = dict(SETTINGS)
    for kv in a.set:
        k, v = kv.split('=', 1)
        settings[k] = json.loads(v) if v[:1] in '0123456789-.[{tfn' and v not in ('soft_l1',) else v
    if a.no_twins:
        settings.update(twin_side=False, twin_jumps=False, twin_same=False, bridge=False)
    t0 = time.time()
    pr = Project(a.project)
    cfg = pr.clip_cfg(a.clip)
    cfg.setdefault('solve', {})
    d = np.load(a.reading or pr.work(a.clip) / 'reader.npz')
    kps = [tuple(k) for k in json.loads(str(d['keypoints']))]
    res = fit_reading(pr, a.clip, cfg, d['uv'], d['conf'], kps, [str(v) for v in d['views']], s=settings)
    cr, P, report = res['creature'], res['params'], res['report']
    path = cr.save(a.clip, P, a.tag, {'reader': True})
    out = cr.export_motion(a.clip, P, pr.out() / f'{a.clip}{a.tag}.npz')
    write_json(pr.work(a.clip) / f'fit{a.tag}-report.json', report)
    print(f'FIT {a.clip}: {len(P)} frames in {time.time() - t0:.1f} s -> {path}, {out}', flush=True)


if __name__ == '__main__':
    main()
