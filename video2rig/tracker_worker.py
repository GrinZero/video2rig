"""Point tracker process (BootsTAPIR, PyTorch). Started by track.py with the torch interpreter, never imported.

    <torch python> tracker_worker.py job.npz out.npz checkpoint.pt cpu|gpu

job: video uint8 [T, S, S, 3] RGB (square), queries float32 [N, 3] (t, y, x; pixel centres at integers).
out: tracks float32 [N, T, 2] (x, y), occlusion and expected_dist logits [N, T], seconds.

Memory: a 5 s clip at 768 x 768 peaks at about 6.5 GB on the card. The process takes at most the card's memory minus
1.2 GB (and never less than 60 % of it; V2R_GPU_MEM_GB sets it by hand): past the card's memory Windows spills into
system RAM and the run crawls without an error, so it fails fast instead, and then tracks the video again at a smaller
size (a 6 GB card tracks at 512 x 512, a little less exact).
"""
import os
import sys
import time
from pathlib import Path

GPU = len(sys.argv) > 4 and sys.argv[4] == 'gpu'
if not GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tapnet.torch import tapir_model  # noqa: E402


def run(model, video, queries, dev):
    """model.forward() in three steps: the video's feature grids once, then the queries in chunks (the refinement holds
    patches of every query in every frame: 1024 queries x 121 frames overflowed the card)."""
    tracks, occ, exp = [], [], []
    chunk = int(os.environ.get('V2R_QUERY_CHUNK', 256))
    p = model.num_pips_iter
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
        grids = model.get_feature_grids(video[None], False, None)
        for a in range(0, len(queries), chunk):
            q = queries[None, a:a + chunk]
            qf = model.get_query_features(video[None], False, q, grids, None)
            tr = model.estimate_trajectories(video.shape[-3:-1], False, grids, qf, q, 64)
            tracks.append(torch.mean(torch.stack(tr['tracks'][p::p]), dim=0)[0].float().cpu())
            occ.append(torch.mean(torch.stack(tr['occlusion'][p::p]), dim=0)[0].float().cpu())
            exp.append(torch.mean(torch.stack(tr['expected_dist'][p::p]), dim=0)[0].float().cpu())
            del tr, qf
        del grids
    return torch.cat(tracks), torch.cat(occ), torch.cat(exp)


def main():
    job = np.load(sys.argv[1])
    torch.set_grad_enabled(False)
    dev = torch.device('cuda' if GPU and torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        cap = float(os.environ.get('V2R_GPU_MEM_GB', 0)) or max(.6 * total, total - 1.2)
        torch.cuda.set_per_process_memory_fraction(min(cap / total, .95))
    else:
        torch.set_num_threads(8)
    model = tapir_model.TAPIR(pyramid_level=1, feature_extractor_chunk_size=8)
    model.load_state_dict(torch.load(sys.argv[3], map_location='cpu'))
    model = model.eval().to(dev)
    frames = job['video']
    side = frames.shape[1]
    queries0 = job['queries'].astype(np.float32)
    t0 = time.time()
    note = ''
    for size in (side, 512, 384):
        if size > side:
            continue
        k = size / side
        try:
            video = torch.from_numpy(frames).to(dev)                      # uint8 on the device, then [-1, 1]
            if size != side:
                video = F.interpolate(video.permute(0, 3, 1, 2).float(), size=(size, size), mode='area').permute(0, 2, 3, 1)
            video = video.half() / 127.5 - 1 if dev.type == 'cuda' else video.float() / 127.5 - 1
            q = queries0.copy()
            q[:, 1:] = (q[:, 1:] + .5) * k - .5                              # pixel centres at integers
            queries = torch.from_numpy(q).to(dev)
            tracks, occ, exp = run(model, video, queries, dev)
            tracks = (tracks + .5) / k - .5
            if size != side:
                note = f', at {size} px (the card is small for {side})'
            break
        except torch.cuda.OutOfMemoryError:
            if dev.type != 'cuda' or size == 384:
                raise
            print(f'the card ran out of memory at {size} px: tracking again at a smaller size', flush=True)
            video = queries = None
            torch.cuda.empty_cache()
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    secs = time.time() - t0
    np.savez_compressed(sys.argv[2], tracks=tracks.numpy(), occlusion=occ.numpy(), expected_dist=exp.numpy(), seconds=secs)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == 'cuda' else 0.0
    print(f'TRACKED {len(queries0)} points, {frames.shape[0]} frames, {secs:.1f} s on {dev.type}'
          + (f', peak {peak:.1f} GB' if peak else '') + note, flush=True)


if __name__ == '__main__':
    main()
