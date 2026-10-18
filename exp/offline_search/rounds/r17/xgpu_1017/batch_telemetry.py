"""Optional metadata-only observer around frozen native batch dispatch.

No tensor copies, synchronization, RNG calls, queue regrouping or forward changes.
This adds Python overhead; enabled/disabled runs are distinct measurement arms.
"""
import copy
import itertools
import json
import os
from pathlib import Path
import runpy
import sys
import threading


def install(module=None, core_class=None):
    if module is None:
        from exp.r17_integration.common import batch_runtime as module
    if core_class is None:
        from openpi.serving.batching_core import BatchingCore as core_class
    if getattr(module, '_xgpu_observed', False):
        raise RuntimeError('batch observer installed twice')
    module._xgpu_observed = True
    local = threading.local()
    sequence = itertools.count()
    native_input = module.BatchedAdapter._input
    def input_(self, value):
        item = native_input(self, value)
        item._xgpu_request = copy.deepcopy(getattr(local, 'request', None))
        return item
    module.BatchedAdapter._input = input_
    native_run = module._StageBatcher._run
    def run(items, call):
        outputs = native_run(items, call)
        members = [getattr(item, '_xgpu_request', None) for item in items]
        identity = str(os.getpid())+':'+str(next(sequence))
        for row, output in enumerate(outputs):
            output._xgpu_batch = dict(batch_id=identity,resident_pid=os.getpid(),size=len(items),row=row,
                                      members=members)
        return outputs
    module._StageBatcher._run = staticmethod(run)
    native_submit = core_class.submit_to_stage
    def submit(self, stage, *args, **kwargs):
        reply = native_submit(self, stage, *args, **kwargs)
        if hasattr(reply, '_xgpu_batch') and hasattr(local, 'receipts'):
            local.receipts.append(dict(stage=stage, **reply._xgpu_batch))
        return reply
    core_class.submit_to_stage = submit
    native_wrap = module.BatchRuntime.wrap_connection_factory
    def wrap(self, factory):
        base = native_wrap(self, factory)
        def observed(*args, **kwargs):
            connection = base(*args, **kwargs)
            native_infer = connection.infer
            def infer(obs):
                local.request = copy.deepcopy(obs.get('__request__'))
                local.receipts = []
                try:
                    result = native_infer(obs)
                    result['__hit_meta__']['xgpu_batch_composition'] = local.receipts
                    return result
                finally:
                    del local.request, local.receipts
            connection.infer = infer
            return connection
        return observed
    module.BatchRuntime.wrap_connection_factory = wrap


def observe_after_stage1(adapter, root, collector=None):
    """Cosmos resets cuDNN flags inside stage1, after its loader receipt.

    Observe its first completed stage1 (normally startup parity). Return the
    identical native object, without changing RNG, flags or model tensors.
    """
    import threading
    native = adapter.run_stage1
    lock, recorded = threading.Lock(), False
    def stage1(*args, **kwargs):
        nonlocal recorded
        result = native(*args, **kwargs)
        if not recorded:
            with lock:
                if not recorded:
                    if collector is None:
                        from inventory import collect
                        report = collect(with_torch=True)
                    else:
                        report = collector()
                    report['scope'] = 'after first native Cosmos stage1; normally startup parity, not a kernel dispatch profile'
                    (Path(root)/'NUMERICS_AFTER_STAGE1.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
                    recorded = True
        return result
    adapter.run_stage1 = stage1


def run(bootstrap, numerics, model):
    from cache_telemetry import install as install_cache
    install_cache()
    # Only pi05 uses the shared stage runtime in these pins. DP/FW already
    # emit native membership ledgers; GR/Cosmos are B1 per replica and their
    # older frozen releases need not contain common.batch_runtime at all.
    if model=='pi05':install()
    # Observe after model loaders have set TF32/cuDNN flags themselves.
    from exp.r17_integration.qualification import model_runtime
    native_load=model_runtime.load
    def load(*args,**kwargs):
        value=native_load(*args,**kwargs)
        from inventory import collect
        report=collect(with_torch=True)
        report['scope']='live resident immediately after native model load; actual attention dispatch is unprofiled'
        report['model']=model
        if '--directory' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--directory')+1]);root.mkdir(parents=True,exist_ok=True)
            (root/'NATIVE_NUMERICS.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
            if model=='cosmos':
                adapter, _provider = value
                observe_after_stage1(adapter,root)
        return value
    model_runtime.load=load
    if numerics:
        from numerics_bootstrap import run as numerical_run
        numerical_run(bootstrap, numerics, model)
    elif bootstrap:
        exec(compile(open(bootstrap).read(), bootstrap, 'exec'), {'__name__': '__main__'})
    else:
        runpy.run_module('exp.r17_integration.ops.formal_server', run_name='__main__')
