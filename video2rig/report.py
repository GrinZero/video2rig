"""Quality numbers of a fit, for you to know where to look (a number never replaces watching the result):
outline match with the video per view, the worst frames, twist about the bone, the biggest joint jump in one frame,
frames with the body under the floor, and the joints that sit at the end of their range in many frames (there the
video asked for more than the body can do, or the camera did not see that part: look at those frames)."""
import numpy as np

from .common import write_json


def at_limit(creature, clip, params, margin_deg=2.0):
    """Per bone, the frames where one of its channels is within margin_deg of its range (the worst six bones)."""
    rig = creature.rig
    lo, hi, _ = creature.bounds(clip)
    m = np.radians(margin_deg)
    near = (params < lo + m) | (params > hi - m)
    hit = {}
    for j, a in rig.channels:
        c = rig.index(rig.names[j], a)
        if np.isfinite(lo[c]) and np.isfinite(hi[c]):
            hit[rig.names[j]] = hit.get(rig.names[j], np.zeros(len(params), bool)) | near[:, c]
    out = {k: int(v.sum()) for k, v in hit.items() if v.any()}
    return dict(sorted(out.items(), key=lambda kv: -kv[1])[:6])


def report(creature, clip, params, path=None, every=2):
    rig, mesh = creature.rig, creature.mesh
    views = creature.views(clip)
    T = len(params)
    iou = np.full((T, len(views)), np.nan)
    for i in range(0, T, every):
        X = rig.skin(params[i], mesh['coords'], creature.idx, creature.wts)
        for k, v in enumerate(views):
            tri, _, _ = creature.rasterize(v['project'](X), v['depth'](X), mesh['tris'], v['size'])
            m, vm = tri >= 0, v['mask'](i)
            iou[i, k] = (m & vm).sum() / max((m | vm).sum(), 1)
    twist = {}
    for j, a in rig.channels:
        if a == 1 and j != rig.root:                 # (the root's three are a rotation vector, not a twist)
            twist[rig.names[j]] = round(float(np.degrees(np.abs(params[:, rig.index(rig.names[j], 1)]).max())), 1)
    rot = params[:, 3:rig.prop0]
    jump = np.degrees(np.abs(np.diff(rot, axis=0))) if T > 1 else np.zeros((1, rot.shape[1]))
    k = np.unravel_index(int(np.argmax(jump)), jump.shape) if jump.size else (0, 0)
    jname = rig.names[rig.channels[k[1]][0]] if jump.size else ''
    under = 0
    low = np.where((mesh['coords'] @ creature.up) < np.percentile(mesh['coords'] @ creature.up, 5))[0]
    low = low[creature.idx[low, 0] < rig.nb]
    for p in params:
        h = rig.skin(p, mesh['coords'][low], creature.idx[low], creature.wts[low]) @ creature.up - creature.floor
        under += bool((h < -.01 * creature.height).any())
    out = {'frames': int(T),
           'outline_match': {v['name']: round(float(np.nanmean(iou[:, k_])), 3) for k_, v in enumerate(views)},
           'worst_frames': {v['name']: [int(i) for i in np.argsort(np.nan_to_num(iou[:, k_], nan=9))[:5]]
                            for k_, v in enumerate(views)},
           'max_twist_deg': dict(sorted(twist.items(), key=lambda kv: -kv[1])[:6]),
           'biggest_jump_deg': {'deg': round(float(jump.max()), 1) if jump.size else 0.0, 'bone': jname,
                                'frame': int(k[0]) + 1},
           'frames_under_floor': int(under),
           'frames_at_joint_limit': at_limit(creature, clip, params)}
    out['warnings'] = warnings(creature, clip, views, iou, out)
    if path is not None:
        write_json(path, out)
    return out


def warnings(creature, clip, views, iou, out, weak=.6):
    """Plain sentences on where the fit cannot be trusted, so a clip can be judged without reading the numbers."""
    msgs = []
    for v, a in zip(views, creature._load(clip)['aligned']):
        if a['frame0_iou'] < .85:
            msgs.append(f"{v['name']} view: the start pose covers frame 0 only {a['frame0_iou']:.2f}; the clip's first "
                        f"frame is not the start image, or start/cameras.json is not the camera it was rendered with")
    for k, v in enumerate(views):
        sampled = np.where(~np.isnan(iou[:, k]))[0]
        low = sampled[iou[sampled, k] < weak]
        if not len(low):
            continue
        runs, start = [], low[0]
        for a, b in zip(low[:-1], low[1:]):
            if b - a > 4:
                runs.append((start, a))
                start = b
        runs.append((start, low[-1]))
        span = ', '.join(f'{a}-{b}' if b > a else f'{a}' for a, b in runs)
        msgs.append(f"{v['name']} view: outline weak (under {weak}) in frames {span}, lowest "
                    f"{np.nanmin(iou[:, k]):.2f}: watch these")
    j = out['biggest_jump_deg']
    if j['deg'] > 45:
        msgs.append(f"{j['bone']} turns {j['deg']:.0f} degrees in one frame at frame {j['frame']}")
    if out['frames_under_floor']:
        msgs.append(f"the body goes under the floor in {out['frames_under_floor']} frames")
    # knees and elbows straighten at every step: their straight end is not a limit worth a warning
    held = [f'{b} ({n})' for b, n in out.get('frames_at_joint_limit', {}).items()
            if n > .25 * out['frames'] and b not in creature.hinge]
    if held:
        msgs.append(f"joints at the end of their range in over a quarter of the frames: {', '.join(held)}: the video "
                    f"asks for more than the range allows (set joints.<bone>.range), or the camera does not see them")
    return msgs or ['none']
