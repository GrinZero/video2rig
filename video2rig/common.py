"""Paths, the project folder and its config.

A project is one folder per creature (see docs/WORKFLOW.md):
    creature.json            settings (docs/CONFIG.md)
    rig/                     rig.json, mesh.npz, texture.png       <- blender/export_rig.py
    start/                   cameras.json, start.png               <- blender/render_start.py
    clips/<clip>.mp4         the Seedance clips
    work/<clip>/             frames' masks, timing, cameras, tracks, fits, review films (cache, can be deleted)
    out/<clip>.npz           the motion for Blender                <- blender/import_motion.py
"""
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1]
THIRD_PARTY = TOOL / 'third_party'
CACHE = Path(os.environ.get('LOCALAPPDATA') or Path.home() / '.cache') / 'video2rig'
CHECKPOINT_URL = 'https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt'


def torch_python():
    """The interpreter that runs the tracker: $V2R_TORCH_PYTHON, else this one if it has PyTorch."""
    env = os.environ.get('V2R_TORCH_PYTHON')
    if env:
        return Path(env)
    if importlib.util.find_spec('torch') is not None:
        return Path(sys.executable)
    raise SystemExit('PyTorch not found: install it into this Python (see README, "Install") or point '
                     'V2R_TORCH_PYTHON at a Python that has it.')


def ffmpeg():
    """The ffmpeg program (review films, bench clips): $V2R_FFMPEG, else the one on PATH."""
    exe = os.environ.get('V2R_FFMPEG') or shutil.which('ffmpeg')
    if not exe or not Path(exe).exists():
        raise SystemExit('ffmpeg not found: install it (ffmpeg.org) or set V2R_FFMPEG to the ffmpeg program')
    return exe


class Project:
    def __init__(self, folder):
        self.root = Path(folder).resolve()
        path = self.root / 'creature.json'
        if not path.exists():
            raise SystemExit(f'{path} not found: start from examples/creature.json (docs/CONFIG.md)')
        self.cfg = json.loads(path.read_text(encoding='utf-8'))
        self.name = self.cfg.get('name', self.root.name)
        self.rig_dir = self.root / self.cfg.get('rig', 'rig')
        self.start_dir = self.root / self.cfg.get('start', 'start')

    def clip_cfg(self, clip):
        """Creature-wide settings overridden by the clip's own (deep merge for 'solve' and 'views')."""
        out = {k: v for k, v in self.cfg.items() if k != 'clips'}
        own = self.cfg.get('clips', {}).get(clip, {})
        for k, v in own.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = {**out[k], **v}
            else:
                out[k] = v
        return out

    def clip_file(self, clip):
        own = self.cfg.get('clips', {}).get(clip, {})
        f = self.root / own.get('file', f'clips/{clip}.mp4')
        if not f.exists():
            raise SystemExit(f'clip not found: {f}')
        return f

    def markers(self, clip):
        """The markers.json of a clip that started from a painted start frame (blender/render_markers.py), or None:
        the clip's "markers" setting (a path, or true for <project>/start-marked), else the "start" image named in
        clips/<clip>.json when a markers.json lies next to it."""
        own = self.cfg.get('clips', {}).get(clip, {})
        m = own.get('markers')
        if m is False:
            return None
        if isinstance(m, str):
            p = (self.root / m)
            return p / 'markers.json' if p.is_dir() else p
        if m is True:
            return self.root / 'start-marked' / 'markers.json'
        meta = self.root / 'clips' / f'{clip}.json'
        if meta.exists():
            start = json.loads(meta.read_text(encoding='utf-8')).get('start')
            if start:
                p = Path(start).parent / 'markers.json'
                if not p.is_absolute():
                    p = self.root / p
                if p.exists():
                    return p
        return None

    def work(self, clip):
        d = self.root / 'work' / clip
        d.mkdir(parents=True, exist_ok=True)
        return d

    def out(self):
        d = self.root / 'out'
        d.mkdir(parents=True, exist_ok=True)
        return d


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=1) + '\n', encoding='utf-8')
