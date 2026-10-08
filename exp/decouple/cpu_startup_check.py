"""Fresh-exec split startup with CPU models and file-backed websocket stubs.

Runs the real supervisor, child CLI, runtime hooks, blueprint and AF_UNIX/memfd
transport. Only the model loader and websocket listener/peer are substituted;
no TCP listener, checkpoint, GPU or remote worker is used.
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import importlib.util
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace


def _environment():
    # Use the mandated main executable, with precisely the selected model
    # venv's site-packages. -S prevents main-venv JAX masking a GR00T failure.
    sys.path.insert(0, os.environ['OPENPI_CPU_STARTUP_SITE'])
    if os.environ['OPENPI_CPU_STARTUP_MODEL']=='groot':
        import builtins
        original, find = builtins.__import__, importlib.util.find_spec
        def no_jax(name, *args, **kwargs):
            if name=='jax' or name.startswith('jax.'):
                raise ImportError('JAX forbidden in GR00T split startup')
            return original(name,*args,**kwargs)
        builtins.__import__ = no_jax
        importlib.util.find_spec = lambda name,*a,**k: None if name=='jax' or name.startswith('jax.') else find(name,*a,**k)


def _install_models():
    import torch
    from openpi.serving import websocket_policy_server as wps
    from openpi.serving import split_runtime

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(action_horizon=10,action_dim=32,pytorch_compile_mode=None)
            self.action_head = SimpleNamespace(num_inference_timesteps=1,dtype=torch.float32,
                                               config=SimpleNamespace(action_horizon=16,action_dim=7))

    class GrootPolicy:
        def __init__(self):
            self.model = Model()
            self.device = 'cpu'
            # Emulate an HF processor referenced by name in the blueprint.
            # Only the GPU loader creates/imports it; a fresh front must use
            # the launcher-provided HF module search path to unpickle it.
            cache = Path(os.environ['HF_MODULES_CACHE'])
            package = cache/'transformers_modules/cpu_probe'
            package.mkdir(parents=True)
            (package.parent/'__init__.py').write_text('')
            (package/'__init__.py').write_text('')
            (package/'processor.py').write_text('class CPUProcessor:\n    pass\n')
            sys.path.append(str(cache))
            processor = importlib.import_module('transformers_modules.cpu_probe.processor')
            self.processor = processor.CPUProcessor()

    module = ModuleType('gr00t.model.policy')
    module.Gr00tPolicy = GrootPolicy
    sys.modules[module.__name__] = module
    from openpi.cache.groot import staged
    # The real runner/coordinator numerical behavior has separate stage proofs.
    staged.GrootStagedRunner = lambda model: SimpleNamespace(model=model)
    split_runtime.make_groot_coordinator = lambda *a,**k: SimpleNamespace()
    if os.environ['OPENPI_CPU_STARTUP_MODEL']=='pi05':
        from openpi.policies import policy_config
        from openpi.serving import batching_core
        def load(*args,**kwargs):
            return SimpleNamespace(_model=Model(),_input_transform=(),_output_transform=(),
                                   _sample_kwargs={},_metadata={})
        policy_config.create_trained_policy = load
        batching_core.get_active_coordinator = lambda: SimpleNamespace()

    class Server:
        def __init__(self,policy,*,host='0.0.0.0',port=8000,metadata=None,concurrent=False,**kwargs):
            self._policy,self._metadata = policy,metadata or {}
            self.host,self.port = host,port
            self._connection_policy_factory = lambda: policy

        def serve_forever(self):
            from openpi.serving.split_service import StageClient
            from openpi.serving.split_socket import read_endpoint
            endpoint = read_endpoint(os.environ['OPENPI_CPU_STARTUP_RUNTIME'])
            assert stat.S_IMODE(endpoint.stat().st_mode)==0o600
            assert not os.environ.get('CUDA_VISIBLE_DEVICES')
            assert self.host=='127.0.0.1' and self._metadata['split']
            model = getattr(self._policy,'_model',getattr(self._policy,'model',None))
            assert not isinstance(model,torch.nn.Module), 'front loaded model weights'
            assert self._connection_policy_factory().__class__.__name__=='DecisionPolicy'
            client = StageClient(endpoint,capacity=2<<20,timeout=5)
            hello = client.call('hello')
            assert torch.equal(hello['probe_tensor'],torch.arange(16))
            client.channel().close()
            root = Path(os.environ['OPENPI_CPU_STARTUP_EVIDENCE'])
            (root/f'front_{self.port}.json').write_text(json.dumps(dict(
                pid=os.getpid(),endpoint=str(endpoint),host=self.host,port=self.port,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                stage_epoch=hello['epoch'],blueprint_kind=self._metadata['model'],
                dynamic_processor_imported=hasattr(self._policy,'processor'),
                jax_imported='jax' in sys.modules,memfd_tensor_equal=True)))
            # Remain owned by the supervisor until its normal shutdown path.
            import signal
            while True:
                signal.pause()

    wps.WebsocketPolicyServer = Server


def _entry():
    """Existing-entrypoint analogue, called only AFTER real role hooks install."""
    import torch
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=23400)
    parser.add_argument('--cache-config',required=True)
    args = parser.parse_args()
    model = os.environ['OPENPI_CPU_STARTUP_MODEL']
    if model=='groot':
        from gr00t.model.policy import Gr00tPolicy
        policy = Gr00tPolicy()
    else:
        from openpi.policies.policy_config import create_trained_policy
        policy = create_trained_policy(None,None)
    WebsocketPolicyServer(policy,port=args.port,concurrent=True,
                          metadata=dict(model=model,probe_tensor=torch.arange(16))).serve_forever()


def _child(argv):
    _install_models()
    # Capture the actual launcher-provided runtime for the stub serving entry.
    os.environ['OPENPI_CPU_STARTUP_RUNTIME'] = argv[argv.index('--runtime-dir')+1]
    from openpi.serving.split import main
    main(argv)


def _supervisor(out,model):
    from openpi.serving import split, replica_proxy
    import websockets.asyncio.server
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    evidence = Path(tempfile.mkdtemp(prefix='run-',dir=out))
    parent = evidence/('long_runtime_parent_'+'x'*80)/('nested_'+'y'*80)
    parent.mkdir(parents=True)
    assert len(os.fsencode(parent))>108
    entry = evidence/'entry.py'
    entry.write_text('from exp.decouple.cpu_startup_check import _entry\n_entry()\n')
    os.environ['OPENPI_CPU_STARTUP_EVIDENCE'] = str(evidence)
    children,logs = [],[]
    popen = subprocess.Popen
    def launch(argv,**kwargs):
        assert argv[1:3]==['-m','openpi.serving.split']
        role = argv[argv.index('--role')+1]
        name = role+'_'+str(len(children))+'.log'
        log = (evidence/name).open('w'); logs.append(log)
        child = popen([sys.executable,'-S','-B','-m','exp.decouple.cpu_startup_check', '--child',*argv[3:]],
                      **kwargs,stdout=log,stderr=subprocess.STDOUT)
        children.append(child)
        return child
    class Complete(Exception):
        pass
    class Peer:
        async def close(self):
            pass
    class Proxy:
        def __init__(self,host,ports):
            assert host=='127.0.0.1'
            self.ports = ports
        async def _open_backend(self,backend,**kwargs):
            if not (evidence/f'front_{backend[1]}.json').is_file():
                raise ConnectionRefusedError()
            return Peer()
        async def prime_metadata(self):
            assert len(list(evidence.glob('front_*.json')))==8
        def handle(self,*args):
            raise AssertionError('no TCP requests in CPU startup check')
    @asynccontextmanager
    async def serve(handler,host,port,**kwargs):
        assert host=='127.0.0.1' and port==23400
        raise Complete()
        yield  # async context manager; never bind any TCP socket
    split.subprocess.Popen = launch
    replica_proxy.ReplicaProxy = Proxy
    websockets.asyncio.server.serve = serve
    try:
        try:
            split.main(['--model',model,'--fronts','8','--port','23400','--host','127.0.0.1',
                        '--arena-mb','2','--startup-timeout','30','--runtime-parent',str(parent),
                        '--log-root',str(evidence/'logs'),'--',str(entry),'--cache-config','stub.yaml'])
        except Complete:
            pass
        else:
            raise AssertionError('split startup did not reach proxy readiness')
    finally:
        for log in logs:
            log.close()
    pointer = json.loads((evidence/'logs/stage_endpoint.json').read_text())
    assert not Path(pointer['directory']).exists(), 'private socket directory leaked'
    assert len(children)==9 and all(child.poll() is not None for child in children)
    rows = [json.loads(p.read_text()) for p in sorted(evidence.glob('front_*.json'))]
    assert {row['endpoint'] for row in rows}=={pointer['endpoint']}
    assert len({row['stage_epoch'] for row in rows})==1
    if model=='groot':
        assert all(not row['jax_imported'] for row in rows)
    report = dict(passed=True,model=model,fronts=8,fresh_children=9,children_reaped=True,
                  runtime_path_bytes=len(os.fsencode(parent)),endpoint=pointer['endpoint'],
                  socket_cleaned=True,fronts_report=rows,evidence=str(evidence),tcp_listeners=0)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CPU_SPLIT_STARTUP_OK '+model)


def check(config,out):
    reports = {}
    for model,case in config['cases'].items():
        site = str(Path(case['python']).parents[1]/'lib/python3.11/site-packages')
        env = {**os.environ,**case['environment'],'CUDA_VISIBLE_DEVICES':'',
               'OPENPI_CPU_STARTUP_SITE':site,'OPENPI_CPU_STARTUP_MODEL':model}
        target = Path(out)/model
        target.mkdir(parents=True,exist_ok=True)
        # TERM lets split's supervisor reap children even if this check hits
        # its outer timeout. A process group is exclusively owned by this test.
        child = subprocess.Popen([sys.executable,'-S','-B','-m','exp.decouple.cpu_startup_check',
                                  '--model',model,'--out',str(target)],env=env,
                                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:
            stdout,stderr = child.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                stdout,stderr = child.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid,signal.SIGKILL)
                stdout,stderr = child.communicate(timeout=5)
            (target/'launcher.log').write_text(stdout+stderr)
            raise TimeoutError(f'{model} CPU split startup timed out; see {target}/launcher.log')
        (target/'launcher.log').write_text(stdout+stderr)
        if child.returncode:
            raise ValueError(f'{model} CPU split startup failed; see {target}/launcher.log:\n'+stderr[-2000:])
        reports[model] = json.loads((target/'report.json').read_text())
    return reports


if __name__=='__main__':
    _environment()
    if sys.argv[1:2]==['--child']:
        _child(sys.argv[2:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--model',choices=('pi05','groot'),required=True)
        parser.add_argument('--out',type=Path,required=True)
        args = parser.parse_args()
        _supervisor(args.out,args.model)
