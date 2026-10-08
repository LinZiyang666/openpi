"""Input-scoped strict numerical proofs, shared across packages and smoke runs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
import socket
import time

from exp.decouple.package_config import REPO, GROOT, canonical, sha, write_json

CACHE = REPO/'exp/decouple/results/gate_cache'
# Qualification never imports the supervisor or allocates its address. Keep
# startup-only fixes outside numerical fingerprints; stage adapters/transport
# remain covered in full.
LAUNCHER_ONLY = {str(REPO/'src/openpi/serving'/name) for name in ('split.py','split_socket.py')}


def file_digest(path, index):
    path = Path(path)
    stat = path.stat()
    identity = [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
    prior = index.get(str(path))
    # This filesystem can give two writes in one clock tick identical stat
    # timestamps. Always rehash small inputs, and newly modified large files.
    old_enough = max(stat.st_mtime,stat.st_ctime) < time.time()-2
    if prior and prior['identity']==identity and stat.st_size >= 64*(1<<20) and old_enough:
        return prior['sha256']
    digest = sha(path)
    after = path.stat()
    if [after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns] != identity:
        raise ValueError(f'qualification input changed while hashing: {path}')
    index[str(path)] = dict(identity=identity,sha256=digest)
    return digest


def input_signatures(config):
    """SHA256 weights once; reuse only when inode/size/mtime/ctime all match."""
    from exp.decouple.recorded import load, observation
    import numpy as np
    index_path = CACHE/'input_files.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    source = {path:digest for path,digest in config['source_sha256'].items() if
              '/src/openpi/' in path and '/conductor/' not in path or
              path.endswith('/exp/decouple/qualify_gpu.py') or path.endswith('/exp/decouple/recorded.py') or
              '/exp/decouple/_vendor/cloudpickle/' in path or
              '/exp/libero_groot/' in path or path.endswith('/openpi_client/msgpack_numpy.py')}
    source = {path:digest for path,digest in source.items() if path not in LAUNCHER_ONLY}
    answer = {}
    for model, case in config['cases'].items():
        checkpoint = Path(case['checkpoint'])
        weight_paths = sorted(checkpoint.glob('*.safetensors'))
        if not weight_paths:
            raise FileNotFoundError(f'no local checkpoint weights: {checkpoint}')
        config_paths = [p for p in (checkpoint/'config.json', checkpoint/'model.safetensors.index.json',
                                    checkpoint/'experiment_cfg/metadata.json') if p.is_file()]
        checkpoint_files = {str(path):file_digest(path,index) for path in weight_paths+config_paths}
        code = dict(source)
        if model == 'groot':
            for directory in (Path(GROOT)/'gr00t',Path(GROOT)/'examples/Libero'):
                code.update({str(p):file_digest(p,index) for p in sorted(directory.rglob('*.py')) if '__pycache__' not in p.parts})
        site = str(Path(case['python']).parents[1]/'lib/python3.11/site-packages')
        dependencies = {dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions(path=[site]) if dist.metadata.get('Name')}
        torch_version = re.search(r"^__version__.*=\s*['\"]([^'\"]+)", (Path(site)/'torch/version.py').read_text(),re.MULTILINE).group(1)
        data = load(model,limit=16)
        recorded = {name:file_digest(data.path/(name+'.npy'),index) for name in ('key_v0','key_v1','rs')}
        fixtures = hashlib.sha256()
        for i in range(16):
            for name, value in sorted(observation(data,i).items()):
                fixtures.update(name.encode())
                fixtures.update(value.encode() if isinstance(value,str) else np.ascontiguousarray(value).tobytes())
        descriptor = dict(model=model,checkpoint=str(checkpoint),checkpoint_files=checkpoint_files,
                          source=code, dependencies=dependencies, torch_version=torch_version,
                          python=file_digest(Path(case['python']).resolve(),index),
                          environment=case['environment'], assets_sha256=config['assets']['sha256'] if model=='pi05' else None,
                          recorded=recorded,fixtures_sha256=fixtures.hexdigest(),rows=16,batch_size=1,
                          config='pi05_libero', noise_seed_start=700)
        for k in (1,2):
            value = dict(descriptor,k=k)
            answer[f'{model}_k{k}_b1'] = dict(sha256=hashlib.sha256(canonical(value)).hexdigest(),descriptor=value)
    write_json(index_path,index)
    return answer


def check_proof(proof, model, k):
    if not (proof.get('model')==model and proof.get('k')==k and proof.get('batch_size')==1 and
            proof.get('rows')==16 and proof.get('passed') and proof.get('donor_order_equal') and
            set(proof.get('fields',{}))=={'action','key:vision_0','key:vision_1','key:robot_state'} and
            all(f.get('bit_exact') for f in proof['fields'].values())):
        raise ValueError('cached gate is not a complete strict bit-exact proof')


class GateCache:
    def __init__(self, root=CACHE):
        self.root = Path(root)

    def put(self, config, key, proof, gpu_identity, *, path=None, owner_attested=False):
        model, k = key.split('_')[0], int(key.split('_')[1][1:])
        check_proof(proof,model,k)
        case = config['cases'][model]
        if proof['checkpoint'] != case['checkpoint']:
            raise ValueError('cached gate checkpoint differs')
        gate_input = config['gate_inputs'][key]
        if proof['torch'] != gate_input['descriptor']['torch_version'] or proof['gpu'] != gpu_identity['name']:
            raise ValueError('cached gate runtime/device differs')
        record = dict(input_sha256=gate_input['sha256'],proof=proof,proof_sha256=hashlib.sha256(canonical(proof)).hexdigest(),
                      gpu_identity=gpu_identity,owner_attested=owner_attested,original_path=str(path) if path else None)
        device = hashlib.sha256(canonical(gpu_identity)).hexdigest()[:12]
        saved = self.root/'proofs'/f'{key}_{gate_input["sha256"]}_{device}.json'
        write_json(saved,proof)
        record['cached_path']=str(saved)
        write_json(self.root/f'{key}_{gate_input["sha256"]}_{device}.json',record)
        return self.result(record)

    @staticmethod
    def result(record):
        return dict(passed=True,path=record['cached_path'],k=record['proof']['k'],batch_size=1,
                    input_sha256=record['input_sha256'],gpu_identity=record['gpu_identity'],cached=True)

    def lookup(self, config, key, gpu_identity=None):
        for path in sorted(self.root.glob(f'{key}_{config["gate_inputs"][key]["sha256"]}_*.json')):
            record = json.loads(path.read_text())
            if hashlib.sha256(canonical(record['proof'])).hexdigest()!=record['proof_sha256']:
                continue
            try:
                check_proof(record['proof'],key.split('_')[0],int(key.split('_')[1][1:]))
            except ValueError:
                continue
            device = record['gpu_identity']
            if gpu_identity is not None:
                if any(device.get(field)!=gpu_identity.get(field) for field in ('host','index','name')):
                    continue
                if device.get('uuid') is not None:
                    if any(device.get(field)!=gpu_identity.get(field) for field in ('uuid','driver')):
                        continue
                elif not record.get('owner_attested'):
                    continue
                else:
                    # The owner's pre-fingerprint proofs name this host/slot
                    # but lack UUID. Bind them to this card on first reuse.
                    record['gpu_identity']=gpu_identity
                    write_json(path,record)
            return self.result(record)
        return None

    def reindex_launcher(self, config, previous):
        """Retain proofs when *only* the excluded launcher scope was removed.

        Compare every other input, including all stage/transport bytes. This
        cannot promote a proof across a changed checkpoint, adapter or device.
        Keep the old entries and a migration receipt for audit.
        """
        migrated = []
        for key, signature in config['gate_inputs'].items():
            old = previous.get('gate_inputs',{}).get(key)
            if old is None or old['sha256']==signature['sha256']:
                continue
            descriptor = json.loads(json.dumps(old['descriptor']))
            if hashlib.sha256(canonical(descriptor)).hexdigest()!=old['sha256']:
                continue
            removed = {path:descriptor['source'].pop(path) for path in LAUNCHER_ONLY if path in descriptor['source']}
            if not removed or descriptor!=signature['descriptor']:
                continue
            result = self.lookup(previous,key)
            if result is None:
                continue
            # Look up the original complete record to preserve attestation.
            records = [json.loads(path.read_text()) for path in self.root.glob(f'{key}_{old["sha256"]}_*.json')]
            record = next(r for r in records if r['cached_path']==result['path'])
            updated = self.put(config,key,record['proof'],record['gpu_identity'],
                               path=record.get('original_path'),owner_attested=record.get('owner_attested',False))
            write_json(self.root/'startup_migrations'/f'{key}_{signature["sha256"]}.json',
                       dict(old_input=old,new_input=signature,removed_launcher_sources=removed,
                            gpu_identity=record['gpu_identity'],proof_sha256=record['proof_sha256'],
                            cached_path=updated['path']))
            migrated.append(key)
        return migrated


def adopt_legacy(config, original, proofs, cache=None):
    """CPU migration of the four owner-attested proofs from this GPU window."""
    cache = cache or GateCache()
    adopted = []
    for key, signature in config['gate_inputs'].items():
        path = Path(proofs)/(key+'.json')
        proof = json.loads(path.read_text())
        for name,digest in signature['descriptor']['source'].items():
            if name in original['source_sha256'] and original['source_sha256'][name]!=digest:
                raise ValueError(f'gate code changed since owner proof: {name}')
            if name not in original['source_sha256'] and Path(name).stat().st_mtime > path.stat().st_mtime:
                raise ValueError(f'external model code changed since owner proof: {name}')
        for name in signature['descriptor']['checkpoint_files']:
            if Path(name).stat().st_mtime > path.stat().st_mtime:
                raise ValueError(f'checkpoint modified since owner proof: {name}')
        identity = dict(host=socket.gethostname(),index=config['gpu_index'],name=proof['gpu'],uuid=None,driver=None)
        cache.put(config,key,proof,identity,path=path,owner_attested=True)
        adopted.append(key)
    return adopted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPO/'exp/decouple/benchmark.local.json')
    parser.add_argument('--import-owner-proofs',type=Path,help='owner-attested GPU result directory (CPU-only migration)')
    parser.add_argument('--original-config',type=Path,default=REPO/'exp/decouple/results/gpu_window_original_config.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.import_owner_proofs:
        print(json.dumps(dict(adopted=adopt_legacy(config,json.loads(args.original_config.read_text()),args.import_owner_proofs)),indent=2))
    else:
        print(json.dumps({key:GateCache().lookup(config,key) for key in config['gate_inputs']},indent=2))


if __name__=='__main__':
    main()
