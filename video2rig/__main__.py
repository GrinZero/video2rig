"""video2rig: a Seedance clip of your own rigged model -> an animation of that rig.

    python -m video2rig run <project> <clip> [--refine] [--cpu]   # everything: frames, joints (or tracks), fit, motion
                                                                  # file; the joint reader when the creature has one
                                                                  # (python -m video2rig.reader.prepare), --old: not
    python -m video2rig check <project> <clip>                    # before the fit: do the panels agree? (seconds)
    python -m video2rig review <project> <clip>                   # comparison film (video | fit | overlay | 3/4)
    python -m video2rig track <project> <clip> [--retrack]        # only the point tracks (cached)
    python -m video2rig solve <project> <clip>                    # only the solve (uses the cached tracks)

<project> is the creature's folder (docs/WORKFLOW.md); the motion lands in <project>/out/<clip>.npz for
blender/import_motion.py.
"""
import argparse
import json
import time

from .common import Project, write_json
from .creature import Creature


def fit_meta(path):
    """What a fit file says about how it was made (Creature.save's meta): the reader road stores str({'reader':
    True}), the old solve a JSON object. {} when it says nothing readable."""
    import ast
    import numpy as np
    d = np.load(path)
    if 'meta' not in d.files:
        return {}
    raw = str(d['meta'])
    for parse in (json.loads, ast.literal_eval):
        try:
            meta = parse(raw)
        except (ValueError, SyntaxError):
            continue
        return meta if isinstance(meta, dict) else {}
    return {}


def review_creature(project, cfg, fit, work, tag, log):
    """The creature a fit was made with. A reader road fit was made with fit.creature_for (road.fit_reading): bones
    the reader sees at both ends, such as a jaw, are fitted too, so the fit has more values per frame and the props
    sit further along. Built the same way here from the reading's keypoints (work/<clip>/reader*.npz), or the jaw
    shows shut and the props are read from the wrong columns."""
    if not fit_meta(fit).get('reader'):
        return Creature(project, cfg)
    import numpy as np
    from .reader.fit import creature_for
    readings = [work / f'reader{tag}.npz', work / 'reader.npz']
    readings += sorted(work.glob('reader*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in readings:
        if path.exists():
            d = np.load(path)
            if 'keypoints' in d.files:
                kps = [tuple(k) for k in json.loads(str(d['keypoints']))]
                log(f'review: a reader road fit, its creature built from the keypoints of {path.name}')
                return creature_for(project, cfg, kps)[0]
    raise SystemExit(f'{fit} came from the joint reader, but {work} holds no reading (reader*.npz) to rebuild its '
                     'creature from: run the clip again')


def main():
    ap = argparse.ArgumentParser(prog='video2rig')
    ap.add_argument('command', choices=['run', 'check', 'track', 'solve', 'review'])
    ap.add_argument('project')
    ap.add_argument('clip')
    ap.add_argument('--cpu', action='store_true', help='track on the CPU (much slower)')
    ap.add_argument('--retrack', action='store_true', help='track again even if cached')
    ap.add_argument('--refine', action='store_true', help='a second pass: re-seed on later frames from the first fit')
    ap.add_argument('--tag', default='', help='suffix of the fit file in work/<clip>/')
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--guide', default='', help="another tool's reading of the clip on the same armature: an npz of "
                    "names, basis (T, bones, 4, 4: pose-bone bases) and heads (armature space), see solve(guide=)")
    ap.add_argument('--old', action='store_true', help='the point tracker and the old solve even when the creature '
                    'has a joint reader (reader/reader.pt)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='override one setting of the older road\'s solve for this run (solve.DEFAULTS), e.g. --set '
                    'w_gait=0; the reader fit\'s settings: python -m video2rig.reader.fit <project> <clip> --set')
    a = ap.parse_args()
    project = Project(a.project)
    if a.command == 'check':
        from .check import check
        res = check(project, a.clip)
        write_json(project.work(a.clip) / 'check.json', res)
        print(f'{res["verdict"]} {a.clip}: body {res.get("body")}, limbs {res.get("limbs")} (share of the height, '
              f'median), frame 0 {res["frame0_outline"]}' + ''.join(f'\n  {m}' for m in res['messages']), flush=True)
        raise SystemExit(1 if res['verdict'] == 'FAIL' else 0)
    cfg = project.clip_cfg(a.clip)
    cfg.setdefault('solve', {})
    for kv in a.set:
        k, v = kv.split('=', 1)
        try:
            cfg['solve'][k] = json.loads(v)
        except json.JSONDecodeError:
            cfg['solve'][k] = v
    if a.rounds is not None:
        cfg['solve']['rounds'] = a.rounds
    if a.iters is not None:
        cfg['solve']['lm_iters'] = a.iters
    t0 = time.time()
    clock = {}
    log = lambda s: print(f'[{time.time() - t0:6.1f}s] {s}', flush=True)
    work = project.work(a.clip)
    if a.command == 'review':
        from .review import review
        fit = work / f'fit{a.tag}.npz'
        if not fit.exists():
            raise SystemExit(f'{fit} not found: run the clip first (python -m video2rig run {a.project} {a.clip})')
        creature = review_creature(project, cfg, fit, work, a.tag, log)
        P = creature.load_params(fit)
        if P.shape[1] != creature.rig.n:
            raise SystemExit(f'{fit}: {P.shape[1]} values per frame, the creature has {creature.rig.n}: the fit was '
                             'made with other settings or another rig; run the clip again')
        log(f'review -> {review(creature, a.clip, [("VIDEO2RIG", P)], work / f"review{a.tag}", work / f"review{a.tag}.mp4")}')
        return
    creature = Creature(project, cfg)
    views = creature.views(a.clip)
    T = len(creature.times(a.clip))
    clock['prepare_s'] = round(time.time() - t0, 1)
    log(f'{a.clip}: {T} frames, {len(views)} view(s), timing {creature._clip[a.clip]["timing"].get("content")}, '
        f'frame 0 outline match after alignment {[r["frame0_iou"] for r in creature._clip[a.clip]["aligned"]]}')
    from .reader.road import reader_path
    if a.command == 'run' and not a.old and reader_path(project) is not None:
        from .reader.road import run as reader_run
        t1 = time.time()
        creature, P, out = reader_run(project, a.clip, cfg, tag=a.tag, log=log, cpu=a.cpu)
        clock['reader_s'] = round(time.time() - t1, 1)
        from .report import report
        rep = report(creature, a.clip, P, work / f'report{a.tag}.json')
        clock['total_s'] = round(time.time() - t0, 1)
        write_json(work / f'clock{a.tag}.json', clock)
        log(f'motion -> {out}')
        log('report ' + json.dumps(rep))
        log('time ' + json.dumps(clock))
        return
    if a.command == 'run':
        log('the older road (point tracker + solve): ' + ('--old' if a.old else
            f'no joint reader for this creature ({project.root / "reader" / "reader.pt"}); '
            f'python -m video2rig.reader.prepare {a.project} --blend <model.blend> makes one (docs/WORKFLOW.md)'))
    from .track import merge, reseed, track
    t1 = time.time()
    tracks = track(creature, a.clip, cfg, device='cpu' if a.cpu else 'gpu', retrack=a.retrack)
    clock['track_s'] = round(time.time() - t1, 1)
    if a.command == 'track':
        log(f'tracks: {len(tracks["tri"])} points')
        return
    from .solve import solve
    t1 = time.time()
    guide = None
    if a.guide:
        import numpy as np
        g = np.load(a.guide)
        names = [str(n) for n in g['names']]
        guide = np.stack([creature.rig.params_from_basis(g['basis'][t], names, g['heads'][t])
                          for t in range(len(g['basis']))])
    fit = solve(creature, a.clip, cfg, tracks, tag=a.tag + ('-pass1' if a.refine else ''), guide=guide)
    if a.refine:
        P1 = creature.load_params(fit)
        more = reseed(creature, a.clip, cfg, P1, fit, device='cpu' if a.cpu else 'gpu', retrack=a.retrack)
        fit = solve(creature, a.clip, cfg, merge(tracks, more), tag=a.tag, init=P1)
    clock['solve_s'] = round(time.time() - t1, 1)
    P = creature.load_params(fit)
    out = creature.export_motion(a.clip, P, project.out() / f'{a.clip}{a.tag}.npz')
    from .report import report
    rep = report(creature, a.clip, P, work / f'report{a.tag}.json')
    clock['total_s'] = round(time.time() - t0, 1)
    write_json(work / f'clock{a.tag}.json', clock)
    log(f'motion -> {out}')
    log('report ' + json.dumps(rep))
    log('time ' + json.dumps(clock))


if __name__ == '__main__':
    main()
