"""CPU witnesses for missing timan108 config assets and sealed local originals."""
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

import yaml

from exp.r17_integration.ops import formal_eval, worker_installs as install


def manifest(files):
    sealed = dict(schema='r17.libero_original_assets.v1',files={
        name:hashlib.sha256(data).hexdigest() for name,data in files.items()})
    return dict(**sealed,digest=hashlib.sha256(json.dumps(sealed,sort_keys=True,separators=(',',':')).encode()).hexdigest())


class WorkerAssetTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'originals'; source.mkdir()
        data={'scenes/original.xml':b'<mujoco/>','textures/original.png':b'original-pixels'}
        for name,content in data.items():
            path=source/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(content)
        return source,manifest(data)

    def writable(self,root):
        for path in root.rglob('*'):
            if path.is_dir() and not path.is_symlink():path.chmod(0o755)

    def test_missing_legacy_asset_path_uses_verified_immutable_originals_and_private_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); originals,document=self.fixture(root)
            package=root/'source/libero/libero'; package.mkdir(parents=True)
            (package/'__init__.py').write_text('# original simulator\n')
            config=root/'shared.yaml'
            config.write_text('assets: /missing/libero/assets\ninit_states: /original/init_files\n')
            before=config.read_bytes()
            store=root/'formal/dependencies'
            dep=install.install_dependencies(root/'source',store,config,
                assets_manifest=document,original_assets=originals)
            assets=Path(dep['assets']); release=Path(dep['asset_release'])
            self.assertEqual(dep['assets_digest'],document['digest'])
            self.assertEqual((Path(dep['private_dependencies'])/'libero/libero/assets').resolve(),assets)
            self.assertFalse(stat.S_IMODE(release.stat().st_mode)&0o222)
            cfg=install.private_config(config,root/'campaign/config',assets=assets)
            own=yaml.safe_load((Path(cfg['path'])/'config.yaml').read_text())
            self.assertEqual(own['assets'],str(assets))
            self.assertEqual(own['init_states'],'/original/init_files')
            self.assertEqual(config.read_bytes(),before)
            # Continued use survives disappearance of the old cache directory.
            reused=install.install_dependencies(root/'source',store,config,
                assets_manifest=document,original_assets='/absent/source')
            self.assertEqual(dep,reused)
            self.writable(root)

    def test_reuse_preserves_marker_and_original_cache_metadata_is_never_copied(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,document=self.fixture(root)
            cache=source/'.cache/huggingface'; cache.mkdir(parents=True)
            (cache/'download.lock').write_text('unrelated active cache lock')
            before=(cache/'download.lock').read_bytes()
            result=install.provision_assets(source,root/'releases',document)
            marker=Path(result['asset_release'])/'ASSETS.json'; stamp=marker.stat().st_mtime_ns
            self.assertEqual(install.provision_assets(source,root/'releases',document),result)
            self.assertEqual(stamp,marker.stat().st_mtime_ns)
            self.assertFalse((Path(result['assets'])/'.cache').exists())
            self.assertEqual((cache/'download.lock').read_bytes(),before)
            self.writable(root)

    def test_changed_or_extra_originals_fail_before_publication(self):
        for extra in (False,True):
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as temp:
                root=Path(temp); source,document=self.fixture(root)
                if extra:(source/'unexpected.xml').write_text('different-originals')
                else:(source/'scenes/original.xml').write_text('changed-original')
                with self.assertRaisesRegex(ValueError,'asset (SHA mismatch|file set differs)'):
                    install.provision_assets(source,root/'releases',document)
                self.assertFalse((root/'releases'/('libero_'+document['digest'])).exists())

    def test_corrupt_existing_release_is_rejected_without_repairing_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,document=self.fixture(root)
            release=install.provision_assets(source,root/'releases',document)
            target=Path(release['assets'])/'scenes/original.xml'
            target.chmod(0o644);target.write_text('corruption')
            with self.assertRaisesRegex(ValueError,'asset SHA mismatch'):
                install.provision_assets(source,root/'releases',document)
            self.assertEqual(target.read_text(),'corruption')
            self.writable(root)

    def test_unsafe_or_unsealed_manifest_is_rejected(self):
        for document in (manifest({'../foreign.xml':b'x'}),dict(manifest({'a.xml':b'x'}),digest='bad')):
            with self.assertRaises(ValueError):install.asset_manifest(document)

    def test_both_worker_lanes_use_the_sealed_asset_manifest(self):
        for node in ('timan107','timan108'):
            with mock.patch.object(formal_eval,'remote',return_value='{}') as remote:
                formal_eval.install_worker_dependencies(node,'/private/deps','/immutable/payload')
            code=remote.call_args.args[1][-1]
            self.assertIn('/immutable/payload/exp/r17_integration/qualification/specs/libero_original_assets.json',code)
            self.assertIn('/home/zixuans8/.cache/libero/assets',code)
            self.assertNotIn('download_assets',code)


if __name__=='__main__':unittest.main()
