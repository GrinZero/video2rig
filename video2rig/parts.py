"""Each limb's own outline in every frame: SAM 2 (Meta's video segmentation model, Apache 2.0) follows the body and
every limb chain (legs, arms, tail) through the clip, started from our own render of frame 0, where we know which
pixel belongs to which limb. The point tracker loses a leg when four legs of the same fur cross; a model that follows
whole shapes keeps them apart, and says which one is in front (a hidden leg's outline shrinks or vanishes).

Optional: without the sam2 package and its checkpoint the tool works as before. The checkpoint is looked for in
$V2R_SAM2_CHECKPOINT, then <sam2 package>/../checkpoints/sam2.1_hiera_base_plus.pt.
Limbs are the lower parts of the limb chains, from the knee or elbow down (or from the middle of a leg whose bones
are not named so): those are thin, cross each other and fool the point tracker; the shoulder blade and thigh are
mostly trunk on screen. Chains with neither (a tail, ears) stay with the body.
Labels per view and frame: 0 nothing, 1 the body, 2 + i limb i.
"""
import os
from pathlib import Path

import cv2
import numpy as np

from .camera import depth as view_depth
from .camera import project

CONFIG = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
CHECKPOINT = 'sam2.1_hiera_base_plus.pt'


def checkpoint():
    path = os.environ.get('V2R_SAM2_CHECKPOINT')
    if path and Path(path).exists():
        return Path(path)
    try:
        import sam2
    except ImportError:
        return None
    for root in (Path(sam2.__file__).resolve().parents[1], Path(sam2.__file__).resolve().parent):
        p = root / 'checkpoints' / CHECKPOINT
        if p.exists():
            return p
    return None


def bone_objects(cr):
    """Object label per bone: 1 the body (trunk, head, tail, upper limbs), 2 + i limb i (a limb chain from its hinge
    down, with everything hanging below it)."""
    rig = cr.rig
    of = np.ones(rig.nb, int)
    names = ['body']
    for chain in cr.limb_chains(as_list=True):
        start = next((i for i, j in enumerate(chain) if rig.names[j] in cr.hinge), None)
        if start is None:
            if not set(chain) & cr.legs:
                continue
            start = len(chain) // 2
        lower = chain[start:]
        names.append(rig.names[lower[0]])
        for j in range(rig.nb):
            p = j
            while p >= 0 and p not in lower:
                p = rig.parents[p]
            if p >= 0:
                of[j] = len(names)
    return of, names


def triangle_objects(cr, of):
    """Each triangle's object: the most common object of its corners' strongest bones (props: the body)."""
    tris = cr.mesh['tris']
    bone = cr.idx[:, 0]
    n = int(of.max()) + 1
    votes = np.zeros((len(tris), n))
    for k in range(3):
        b = bone[tris[:, k]]
        ok = b < cr.rig.nb
        np.add.at(votes, (np.where(ok)[0], of[b[ok]]), 1)
    votes[:, 1] += 1e-3
    return votes.argmax(1)


def start_labels(cr, clip, tri_obj):
    X0 = cr.rig.skin(cr.p0(clip), cr.mesh['coords'], cr.idx, cr.wts)
    out = []
    for v in cr.views(clip):
        tri, _, _ = cr.rasterize(project(v, X0), view_depth(v, X0), cr.mesh['tris'], v['size'])
        lab = np.zeros(tri.shape, np.uint8)
        lab[tri >= 0] = tri_obj[tri[tri >= 0]]
        out.append(lab)
    return out


def segment(cr, clip, log=print):
    """Label images per view (T, H, W) uint8, cached in work/<clip>/parts.npz; None without SAM 2."""
    work = cr.project.work(clip)
    path = work / 'parts.npz'
    of, names = bone_objects(cr)
    if path.exists():
        d = np.load(path, allow_pickle=True)
        if list(d['names']) == names:
            return [d[f'view{k}'] for k in range(len(cr.views(clip)))], of, names
    ck = checkpoint()
    if ck is None or len(names) < 2:
        return None, of, names
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    tri_obj = triangle_objects(cr, of)
    starts = start_labels(cr, clip, tri_obj)
    pred = build_sam2_video_predictor(CONFIG, str(ck), device='cuda' if torch.cuda.is_available() else 'cpu')
    T = len(cr.times(clip))
    labels = []
    for vi, v in enumerate(cr.views(clip)):
        fdir = work / f'_sam2_frames_{vi}'
        fdir.mkdir(exist_ok=True)
        for i in range(T):
            cv2.imwrite(str(fdir / f'{i:05d}.jpg'), v['frame'](i), [cv2.IMWRITE_JPEG_QUALITY, 95])
        H, W = starts[vi].shape
        best = np.full((T, H, W), 0.0, np.float32)
        lab = np.zeros((T, H, W), np.uint8)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            state = pred.init_state(video_path=str(fdir), offload_video_to_cpu=True)   # frames in RAM: several clips share the card
            for k in range(1, len(names) + 1):
                m0 = starts[vi] == k
                if m0.sum() > 20:
                    pred.add_new_mask(state, frame_idx=0, obj_id=k, mask=m0)
            for t, ids, logits in pred.propagate_in_video(state):
                lg = logits[:, 0].float().cpu().numpy()
                for n_, k in enumerate(ids):
                    win = lg[n_] > np.maximum(best[t], 0.0)   # each pixel to the most confident object
                    lab[t][win] = k
                    best[t][win] = lg[n_][win]
        labels.append(lab)
        for f in fdir.glob('*.jpg'):
            f.unlink()
        fdir.rmdir()
        log(f'parts: {v["name"]} view segmented ({len(names)} objects)')
    np.savez_compressed(path, names=np.array(names), **{f'view{k}': lab for k, lab in enumerate(labels)})
    return labels, of, names
