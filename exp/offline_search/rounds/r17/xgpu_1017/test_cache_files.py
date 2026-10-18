import json
from pathlib import Path
import unittest
import uuid
from artifacts import digest,write_json
from cache_files import checked_file,verify


class CacheFilesTest(unittest.TestCase):
    def test_missing_cache_receipt_is_unknown_and_mismatch_is_rejected(self):
        from concurrent_compare import cache_files_attestation
        root=Path(__file__).parent/'test_artifacts'/('cache_audit_'+uuid.uuid4().hex);root.mkdir(parents=True)
        write_json(root/'RUN.json',dict(pin=dict(identity='pin')))
        self.assertFalse(cache_files_attestation([root])['qualified'])
        value=dict(identity='file_identity',status='PASS',serving_pin='pin')
        write_json(root/'CACHE_FILES_VERIFICATION.json',dict(status='PASS',before=value,after=value))
        self.assertTrue(cache_files_attestation([root])['qualified'])
        write_json(root/'CACHE_FILES_VERIFICATION.json',dict(status='PASS',before=value,after=dict(value,identity='changed')))
        with self.assertRaisesRegex(ValueError,'cache byte verification failed'):cache_files_attestation([root])

    def test_relocated_file_bytes_and_binding_are_enforced(self):
        root=Path(__file__).parent/'test_artifacts'/('cache_files_'+uuid.uuid4().hex)
        root.mkdir(parents=True);data=root/'data';data.mkdir();file=data/'fit.bin';file.write_bytes(b'frozen fit')
        bundle=root/'bundle';bundle.mkdir();write_json(bundle/'PIN.json',{'identity':'serving_pin'})
        body=dict(schema='xgpu.cache.files.v1',serving_pin='serving_pin',
            files={'/data/mw_finetune/eval/fit.bin':checked_file(file)})
        write_json(bundle/'CACHE_FILES.json',dict(body,identity=digest(body)))
        mapping=[('/data/mw_finetune/eval',str(data))]
        self.assertEqual(verify(bundle,mapping)['status'],'PASS')
        file.write_bytes(b'different!')
        with self.assertRaisesRegex(ValueError,'bytes differ'):verify(bundle,mapping)
        file.write_bytes(b'frozen fit');write_json(bundle/'PIN.json',{'identity':'another_pin'})
        with self.assertRaisesRegex(ValueError,'seal/PIN changed'):verify(bundle,mapping)

    def test_changed_seal_and_outside_paths_are_rejected(self):
        root=Path(__file__).parent/'test_artifacts'/('cache_files_'+uuid.uuid4().hex);root.mkdir(parents=True)
        write_json(root/'PIN.json',{'identity':'pin'})
        body=dict(schema='xgpu.cache.files.v1',serving_pin='pin',files={})
        doc=dict(body,identity=digest(body));doc['files']={'/outside/forbidden':{}}
        write_json(root/'CACHE_FILES.json',doc)
        with self.assertRaisesRegex(ValueError,'seal/PIN changed'):verify(root)
        body['files']=doc['files'];write_json(root/'CACHE_FILES.json',dict(body,identity=digest(body)))
        with self.assertRaisesRegex(ValueError,'unsafe cache dependency'):verify(root)


if __name__=='__main__':unittest.main()
