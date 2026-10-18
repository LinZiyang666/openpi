"""CPU regression tests against each actual pinned frozen mps.py.

No Session.start, CUDA import/initialization, nvidia-smi or daemon command.
Small reservation fixtures stay inside the study directory; none are deleted.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

from concurrent_server import mps_directory_plan, mps_names, reserve_mps_directory


WORK = Path(__file__).resolve().parent
GPU_UUID = 'GPU-00000000-1111-2222-3333-444444444444'
MPS_RELATIVE = 'exp/r17_integration/models/fastwam/ops/mps.py'


def frozen_paths(runtime):
    # Execute the unmodified frozen source, not a copied path formula. The only
    # imported helper is formal_process; paths itself uses stdlib globals only.
    with patch.object(sys, 'path', [str(runtime), *sys.path]):
        return runpy.run_path(str(Path(runtime) / MPS_RELATIVE))['paths']


class MpsPathsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = json.loads((WORK/'concurrent/bundles/dp/source_campaign.json').read_text())
        cls.paths = staticmethod(frozen_paths(source['runtime']))

    def test_actual_frozen_paths_every_model_profile_and_host(self):
        rows = []
        for bundle in sorted((WORK/'concurrent/bundles').iterdir()):
            source = json.loads((bundle/'source_campaign.json').read_text())
            path_builder = frozen_paths(source['runtime'])
            module = Path(source['runtime']) / MPS_RELATIVE
            for host, base in [('4090', WORK), ('h100', Path('/home/exouser/xgpu_1017')),
                               ('a100', Path('/home/exouser/xgpu_1017'))]:
                with self.subTest(bundle=bundle.name, host=host):
                    plan = mps_directory_plan(base, GPU_UUID, path_builder, 'Z9')
                    actual = path_builder(plan['campaign_directory'], GPU_UUID)
                    self.assertEqual(plan['pipe_directory'], str(actual[1]))
                    self.assertEqual(plan['pipe_bytes'], len(os.fsencode(actual[1])))
                    self.assertLessEqual(plan['pipe_bytes'], 85)
                    self.assertEqual(plan['pipe_bytes'], 85 if host == '4090' else 39)
                    rows.append(dict(model=source['model'], profile=bundle.name, host=host,
                        frozen_module=str(module), frozen_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                        **plan))
        self.assertEqual({r['model'] for r in rows}, {'pi05', 'groot', 'dp', 'fastwam', 'cosmos'})
        self.assertEqual(len(rows), 27)
        (WORK/'evidence/mps_path_CPU_matrix.json').write_text(json.dumps(rows, indent=2)+'\n')

    def test_original_4090_failure_and_exact_boundary(self):
        with self.assertRaisesRegex(ValueError, 'pipe path too long'):
            self.paths(WORK/'m123', GPU_UUID)
        self.assertEqual(mps_directory_plan(WORK, GPU_UUID, self.paths, 'Z9')['pipe_bytes'], 85)
        with self.assertRaisesRegex(ValueError, 'pipe path too long'):
            mps_directory_plan(Path('/')/('a'*69), GPU_UUID, self.paths, 'Z9')

    def test_counts_bytes_including_resolved_unicode_path(self):
        self.assertEqual(mps_directory_plan(Path('/')/('é'*34), GPU_UUID, self.paths, 'Z9')['pipe_bytes'], 85)
        with self.assertRaisesRegex(ValueError, 'pipe path too long'):
            mps_directory_plan(Path('/')/('é'*35), GPU_UUID, self.paths, 'Z9')

    def test_short_names_are_unique_and_safe(self):
        names = mps_names()
        self.assertEqual(len(names), 62**2)
        self.assertEqual(len(set(names)), len(names))
        self.assertTrue(all(len(n) == 2 and n.isascii() and n.isalnum() for n in names))
        for bad in ('..', '/a', 'abc', 'éa', 'a_'):
            with self.assertRaises(ValueError):
                mps_directory_plan(WORK, GPU_UUID, self.paths, bad)

    def test_reservation_never_reuses_old_directory(self):
        first = reserve_mps_directory(WORK, GPU_UUID, self.paths)
        occupied = Path(first['campaign_directory'])
        sentinel = occupied/'CPU_TEST_ONLY.json'
        sentinel.write_text('{"retained": true}\n')
        with patch('concurrent_server.mps_names', return_value=[occupied.name, *mps_names()]):
            second = reserve_mps_directory(WORK, GPU_UUID, self.paths)
        self.assertNotEqual(first['campaign_directory'], second['campaign_directory'])
        self.assertEqual(sentinel.read_text(), '{"retained": true}\n')
        for plan in (first, second):
            self.assertEqual(Path(plan['campaign_directory']).stat().st_mode & 0o777, 0o700)
            self.assertFalse(Path(plan['root']).exists())  # Test never starts MPS.
        with patch('concurrent_server.mps_names', return_value=[occupied.name]):
            with self.assertRaisesRegex(RuntimeError, 'namespace exhausted'):
                reserve_mps_directory(WORK, GPU_UUID, self.paths)

    def test_atomic_reservation_retries_a_competing_creation(self):
        names = [n for n in mps_names() if not os.path.lexists(WORK/n)][:2]
        native_mkdir = Path.mkdir
        def racing_mkdir(path, *args, **kwargs):
            if path == WORK/names[0]:
                native_mkdir(path, *args, **kwargs)  # Simulate the other launcher winning.
            return native_mkdir(path, *args, **kwargs)
        with patch('concurrent_server.mps_names', return_value=names), patch.object(Path, 'mkdir', racing_mkdir):
            plan = reserve_mps_directory(WORK, GPU_UUID, self.paths)
        self.assertEqual(Path(plan['campaign_directory']).name, names[1])

    def test_overlong_root_fails_before_any_directory_creation(self):
        with patch.object(Path, 'mkdir') as mkdir:
            with self.assertRaisesRegex(ValueError, 'pipe path too long'):
                reserve_mps_directory(WORK/'evidence', GPU_UUID, self.paths)
            mkdir.assert_not_called()


if __name__ == '__main__':
    unittest.main()
