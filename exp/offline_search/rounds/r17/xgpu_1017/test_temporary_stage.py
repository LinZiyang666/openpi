import hashlib
import json
from pathlib import Path
import socket
import unittest
import uuid
from unittest.mock import patch

from pi05_checkpoint_view import create,verify,digest
from temporary_h100_stage import select_port,module,daemon


def directory():
    p=Path(__file__).parent/'test_artifacts'/('temporary_stage_'+uuid.uuid4().hex)
    p.mkdir(parents=True)
    return p  # retain fixtures, no deletion


class Tests(unittest.TestCase):
    def test_reserved_unbound_campaign_ports_and_old_port_rejected(self):
        audit={'local':dict(listeners=[],active_campaigns={'running.json':dict(ports=[45123])}),
               'h100':dict(listeners=[45124],active_campaigns={'remote.json':dict(ports=[45125])})}
        for port in (23177,23118,45123,45124,45125):
            with self.assertRaises(ValueError):select_port(audit,port)

    def test_bind_collision_rejected_even_if_snapshot_stale(self):
        with socket.socket() as sock:
            sock.bind(('0.0.0.0',0));port=sock.getsockname()[1]
            with self.assertRaises(ValueError):select_port({},port)

    def test_module_exports_exact_files_and_rejects_symlink(self):
        root=directory();base=root/'source';base.mkdir();sub=base/'nested';sub.mkdir()
        one=sub/'one.txt';one.write_text('one');(base/'private.txt').write_text('excluded')
        result=module(root,'test',base,[one],'/unused')
        text=Path(result['include']).read_text()
        self.assertEqual(text,'/nested/\n/nested/one.txt\n')
        link=base/'link';link.symlink_to(one)
        with self.assertRaises(ValueError):module(root,'bad',base,[link],'/unused')

    def test_daemon_closed_when_transfer_body_fails(self):
        root=directory();source=root/'payload';source.write_text('test')
        export=module(root,'test',root,[source],'/unused')
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with self.assertRaisesRegex(RuntimeError,'synthetic transfer failure'):
            with daemon(root,port,[export]):raise RuntimeError('synthetic transfer failure')
        self.assertTrue(json.loads((root/'DAEMON_CLOSED.json').read_text())['closed'])
        with socket.socket() as sock:sock.bind(('0.0.0.0',port))
        config=(root/'rsyncd.conf').read_text()
        self.assertIn('hosts deny = *',config);self.assertIn('read only = yes',config)
        self.assertIn('exclude = *',config)

    def manifest_fixture(self):
        root=directory();source=root/'shared';small=root/'sealed';source.mkdir();small.mkdir()
        (source/'model.safetensors').write_bytes(b'same weight bytes')
        (source/'download.metadata').write_bytes(b'host-specific metadata')
        (small/'download.metadata').write_bytes(b'sealed metadata')
        hashes={'model.safetensors':hashlib.sha256((source/'model.safetensors').read_bytes()).hexdigest(),
                'download.metadata':hashlib.sha256((small/'download.metadata').read_bytes()).hexdigest()}
        manifest=dict(content_digests=hashes,tokenizer={'sha256':'test_tokenizer'})
        manifest['checkpoint_identity']=digest(dict(config='pi05_metaworld',files=hashes,tokenizer='test_tokenizer'))
        return root,source,small,manifest

    def test_private_view_fixes_identity_without_altering_shared_metadata(self):
        root,source,small,manifest=self.manifest_fixture();view=root/'view'
        receipt=create(view,source,small,manifest,mib_per_second=0)
        self.assertTrue(receipt['weight_hardlinked'])
        self.assertEqual((source/'download.metadata').read_bytes(),b'host-specific metadata')
        self.assertEqual((view/'download.metadata').read_bytes(),b'sealed metadata')
        self.assertEqual(verify(view,manifest),receipt)
        (view/'extra').write_text('must fail frozen identity')
        with self.assertRaises(ValueError):verify(view,manifest)

    def test_wrong_weights_fail_without_mutating_shared_files(self):
        root,source,small,manifest=self.manifest_fixture()
        (source/'model.safetensors').write_bytes(b'wrong checkpoint')
        with self.assertRaises(ValueError):create(root/'view',source,small,manifest,mib_per_second=0)
        self.assertEqual((source/'model.safetensors').read_bytes(),b'wrong checkpoint')
        self.assertFalse((root/'view_receipt.json').exists())

    def test_actual_frozen_identity_formula(self):
        path=Path(__file__).parent/'concurrent/bundles/pi05/assets/model_manifest.json'
        m=json.loads(path.read_text())
        self.assertEqual(digest(dict(config='pi05_metaworld',files=m['content_digests'],tokenizer=m['tokenizer']['sha256'])),m['checkpoint_identity'])

    def test_h100_run_without_private_view_fails_before_any_subprocess(self):
        import run_concurrent
        root=Path(__file__).parent
        args=['run_concurrent.py','--host','h100','--model','pi05','--work',str(root),'--run']
        with patch('sys.argv',args),patch.object(run_concurrent.subprocess,'call') as spawn:
            with self.assertRaises(SystemExit) as error:run_concurrent.main()
            self.assertEqual(error.exception.code,2);spawn.assert_not_called()

    def test_private_checkpoint_is_explicit_relocation_in_plan(self):
        import run_concurrent
        root=Path(__file__).parent;view='/data/mw_finetune/xgpu_1017/pi05_ckpt_4008289bc944'
        args=['run_concurrent.py','--host','h100','--model','pi05','--work',str(root),'--checkpoint-view',view]
        with patch('sys.argv',args),patch.object(run_concurrent.subprocess,'call',return_value=0) as spawn:
            with self.assertRaises(SystemExit):run_concurrent.main()
            command=spawn.call_args.args[0]
            self.assertIn('/data/ckpt/pi05_metaworld_rlinf='+view,command)
            self.assertNotIn('--run',command)


if __name__=='__main__':unittest.main()
