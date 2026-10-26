"""Portable training handoff. Uses only the standard library until training starts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def source_bundle(output):
    """Create the source upload without requiring a GitHub publication or a git executable."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ['train_colab.py', 'requirements-colab.txt', 'requirements.txt', 'README.md', 'LICENSE']
    files = [ROOT / name for name in names]
    for name in ('video2rig', 'blender', 'third_party', 'docs', 'examples', 'colab', 'tests'):
        files.extend(p for p in (ROOT / name).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc')
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files):
            if path.resolve() == output or not path.resolve().is_relative_to(ROOT):
                raise ValueError(f'Invalid source path: {path}')
            bundle.write(path, Path('video2rig') / path.relative_to(ROOT))
    print(f'Source bundle: {output}', flush=True)


def validate(project):
    project = Path(project).resolve()
    cfg = json.loads((project / 'creature.json').read_text(encoding='utf-8'))
    folders = [project / cfg.get('rig', 'rig'), project / cfg.get('start', 'start'), project / 'reader']
    for folder in folders:
        if not folder.resolve().is_relative_to(project):
            raise ValueError('Training bundles require rig/start/reader paths inside the project')
    required = [folders[0] / 'rig.json', folders[0] / 'mesh.npz', folders[1] / 'cameras.json',
                folders[2] / 'poses.npz', folders[2] / 'visible.npz', folders[2] / 'renders' / 'labels.npz']
    for path in required:
        if not path.is_file():
            raise ValueError(f'Missing {path}; first run reader.prepare with --skip-train')
    cameras = json.loads(required[2].read_text(encoding='utf-8'))
    for view in cameras['views']:
        folder = folders[2] / 'renders' / view['name']
        if not folder.resolve().is_relative_to(project) or not list(folder.glob('*.jpg')):
            raise ValueError(f'Missing render strips: {folder}')
    return project, folders


def pack(project, output):
    project, folders = validate(project)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {project / 'creature.json'}
    for folder in folders:
        files.update(p for p in folder.rglob('*') if p.is_file()
                     and p.suffix.lower() not in {'.pt', '.log', '.zip'})
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if path.resolve() == output or not path.resolve().is_relative_to(project):
                raise ValueError(f'Bundle contains an external link or its own output: {path}')
            archive.write(path, Path('project') / path.relative_to(project))
    print(f'Training bundle: {output}', flush=True)


def unpack(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'Use an empty destination: {destination}')
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or '\\' in member.filename:
                raise ValueError(f'Unsafe archive path: {member.filename}')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symlinks are not supported in training bundles')
        bundle.extractall(destination)
    validate(destination / 'project')
    print(destination / 'project', flush=True)


def train(a):
    project, _ = validate(a.project)
    if a.install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-colab.txt')],
                       check=True)
    cmd = [sys.executable, '-u', '-m', 'video2rig.reader.train', str(project),
           '--steps', str(a.steps), '--minutes', str(a.minutes), '--batch', str(a.batch),
           '--seed', str(a.seed)]
    if a.init:
        cmd += ['--init', str(Path(a.init).resolve())]
    subprocess.run(cmd, cwd=ROOT, check=True)
    output = Path(a.result).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in ('reader.pt', 'reader-train.json'):
            path = project / 'reader' / name
            bundle.write(path, Path('reader') / name)
    print(f'Result: {output}\nExtract into your original creature project, then run python -m video2rig run ...',
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('source', help='create the source ZIP for Colab without publishing to GitHub')
    p.add_argument('--output', default='work/video2rig-colab.zip')
    p = commands.add_parser('pack', help='bundle locally prepared renders for Colab')
    p.add_argument('project')
    p.add_argument('--output', default='work/training.zip')
    p = commands.add_parser('unpack', help='extract a training bundle into an empty directory')
    p.add_argument('archive')
    p.add_argument('destination')
    p = commands.add_parser('train', help='one command: train with existing Python entrypoint and export results')
    p.add_argument('project')
    p.add_argument('--install', action='store_true', help='install training dependencies (keeps existing torch)')
    p.add_argument('--steps', type=int, default=9000)
    p.add_argument('--minutes', type=float, default=8)
    p.add_argument('--batch', type=int, default=4, help='conservative T4 default; lower to 2 if out of memory')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init', default='', help='warm start, not an exact optimizer/scheduler resume')
    p.add_argument('--result', default='work/reader-result.zip')
    a = parser.parse_args()
    try:
        if a.command == 'source':
            source_bundle(a.output)
        elif a.command == 'pack':
            pack(a.project, a.output)
        elif a.command == 'unpack':
            unpack(a.archive, a.destination)
        else:
            if a.batch < 1 or a.steps < 0 or a.minutes <= 0:
                parser.error('batch/minutes must be positive; steps must be nonnegative')
            train(a)
    except (ValueError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
