import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launcher', ROOT / 'train_colab.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

class ColabTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        (root / 'creature.json').write_text(json.dumps({'rig': 'custom-rig', 'start': 'start'}))
        for name in ('custom-rig/rig.json', 'custom-rig/mesh.npz', 'start/cameras.json',
                     'reader/poses.npz', 'reader/visible.npz', 'reader/renders/labels.npz',
                     'reader/renders/front/0000.jpg'):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({'views': [{'name': 'front'}]}) if name.endswith('cameras.json') else 'fixture')
        return root

    def test_bundle_roundtrip_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            project = self.fixture(d / 'creature')
            (project / 'reader/reader.pt').write_text('old weights')
            launcher.pack(project, d / 'data.zip')
            launcher.unpack(d / 'data.zip', d / 'extracted')
            launcher.validate(d / 'extracted/project')
            self.assertFalse((d / 'extracted/project/reader/reader.pt').exists())

    def test_missing_visibility_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.fixture(Path(tmp) / 'creature')
            (p / 'reader/visible.npz').unlink()
            with self.assertRaisesRegex(ValueError, 'visible.npz'):
                launcher.validate(p)

    def test_traversal_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ('../escape', '/absolute', 'a\\b', 'symlink'):
                with zipfile.ZipFile(d / 'bad.zip', 'w') as z:
                    item = zipfile.ZipInfo(name)
                    if name == 'symlink':
                        item.external_attr = 0o120777 << 16
                    z.writestr(item, 'bad')
                with self.assertRaises(ValueError):
                    launcher.unpack(d / 'bad.zip', d / 'unpacked')

    def test_train_uses_existing_module_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p = self.fixture(d / 'creature')
            for name in ('reader.pt', 'reader-train.json'):
                (p / 'reader' / name).write_text('result')
            args = SimpleNamespace(project=p, install=False, steps=2, minutes=8, batch=1,
                                   seed=3, init='', result=d / 'result.zip')
            with patch.object(launcher.subprocess, 'run') as run:
                launcher.train(args)
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[2:4], ['-m', 'video2rig.reader.train'])
            self.assertEqual(cmd[cmd.index('--batch')+1], '1')
            with zipfile.ZipFile(args.result) as z:
                self.assertEqual(set(z.namelist()), {'reader/reader.pt', 'reader/reader-train.json'})

    def test_precision_native_bf16_only(self):
        tree = ast.parse((ROOT / 'video2rig/reader/net.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reader_autocast')
        for available, supported in ((False, False), (True, False), (True, True)):
            from unittest.mock import Mock
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available,
                is_bf16_supported=lambda: supported), bfloat16='bf16', autocast=Mock(return_value=nullcontext()))
            scope = {'torch': torch}
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<precision>', 'exec'), scope)
            with scope['reader_autocast']():
                pass
            self.assertEqual(torch.autocast.call_count, int(available and supported))

    def test_notebook_python_cells_compile(self):
        nb = json.loads((ROOT / 'colab/train_reader.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), '<notebook>', 'exec')

if __name__ == '__main__':
    unittest.main()
