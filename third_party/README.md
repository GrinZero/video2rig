# Vendored code

| Folder | Source | Commit | Licence |
| --- | --- | --- | --- |
| `tapnet/torch/` | github.com/google-deepmind/tapnet (`tapnet/torch`: BootsTAPIR / TAPIR in PyTorch) | 730cda1c730877cfedbe01bf87fb1cadb78a565d | Apache 2.0 (`tapnet/LICENSE`; the checkpoints are Apache 2.0 too) |
| `einshape/` | github.com/google-deepmind/einshape (only the engine, used by `tapnet/torch/utils.py`) | d91ab136da3d97f4ba2b4582531cdc2dfd9fab0a | Apache 2.0 (`einshape/LICENSE`) |
| `tree.py` | our stand-in for dm-tree's `map_structure` (only what `tapir_model.py` calls) | - | ours |

`tapnet/__init__.py` is empty here: upstream it lazily imports the JAX side, which we do not use.

## Local changes to `tapnet/torch/tapir_model.py`

All marked `LOCAL PATCH (video2rig)`:

- `_unit_norm`: the feature grids are L2-normalised chunk by chunk and kept in the input's precision (fp16 under
  autocast). Upstream normalised all frames at once in fp32.
- `get_feature_grids`: the extra convolutions run per chunk of frames too, and the grids are filled in place
  (upstream kept a list of chunks and concatenated them: two copies at the peak).

`video2rig/tracker_worker.py` runs `forward()` in its three steps: the feature grids once, then the queries in
chunks of 256 (the refinement holds patches of every query in every frame; 1024 queries x 121 frames overflowed).

Why: on an 854x480 clip of 97 frames upstream filled the laptop's 12 GB card, and Windows then spills into system
RAM and the run crawls (10+ minutes instead of seconds). With the patches: peak 5.4-6.4 GB, about 1 min for
2400 points over 97 frames at 768x768 (alone on the card).

## Known trap

The torch model misplaces x on non-square input: at 856x480 the query frame itself came back 50-70 px off. The tool
pads every frame to a square with its own border before tracking (`video2rig/track.py`).

The checkpoint (`bootstapir_checkpoint_v2.pt`, 219 MB) is not in the repo; `video2rig/track.py` downloads it to
`%LOCALAPPDATA%\video2rig\` on first use.
