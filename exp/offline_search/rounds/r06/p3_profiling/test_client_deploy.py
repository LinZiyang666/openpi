"""CPU-only client packaging/dispatch/collection checks. Never invoke tether."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from .build_client_bundle import build, HERE, MODULES
from .client_compat import zip_compat, DeferredSourceLoader
from .client_plan import resolve
from .collect_client import unpack
from .install_client_payload import install
from .test_matrix import PREFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke-run", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=HERE/'results/client_deploy_tests.json')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    bundle = a.out / "bundle"
    manifest = build(bundle, a.smoke_run)
    island = a.out / "island"
    (island/"os_cl").mkdir(parents=True)
    stock_bytes = b"#!/bin/bash\n# stock is never run by this test\n"
    (island/"os_cl/run_arm.sh").write_bytes(stock_bytes)
    (island/"exp").mkdir()
    old_marker = b'"""preexisting marker must be preserved"""\n'
    (island/"exp/__init__.py").write_bytes(old_marker)
    install(bundle, island)
    assert (island/"os_cl/run_arm.sh").read_bytes() == stock_bytes
    assert (island/"os_cl/run_arm.stock.sh").read_bytes() == stock_bytes
    assert (island/"exp/__init__.py").read_bytes() == old_marker
    code = '''import sys, runpy
from pathlib import Path
island, repo = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path[:] = [str(island)] + [x for x in sys.path if Path(x or ".").resolve() not in (repo, repo/"src", repo/"packages/openpi-client/src")]
sys.argv = ["client_preflight"]
runpy.run_module("exp.offline_search.rounds.r06.p3_profiling.client_preflight", run_name="__main__")
for name, module in list(sys.modules.items()):
 if name.startswith("exp.offline_search"):
  assert str(island) in module.__file__, (name, module.__file__)
'''
    check = subprocess.run(PREFIX+["-c", code, str(island), str(HERE.parents[4])], check=True, capture_output=True, text=True)
    (a.out/"isolated_import.json").write_text(check.stdout)
    corrupt = bundle/"payload"/manifest['files'][0]['relative_path']
    saved = corrupt.read_bytes()
    corrupt.write_bytes(saved+b"# bad\n")
    try:
        install(bundle, island)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt bundle accepted")
    corrupt.write_bytes(saved)
    names = (a.smoke_run/"arm_names.txt").read_text().splitlines()
    plans = [resolve(a.smoke_run, n, {}) for n in names]
    assert all(p['client_env']['P3_ENV_SEED']=='603' and '/p3_client/' in p['telemetry_remote'] for p in plans)
    over = resolve(a.smoke_run, names[0], dict(P3_ENV_SEED="605", P3_SNAPSHOT_EVERY="3", P3_SNAPSHOT_P="0.5"))
    assert over['client_env']['P3_ENV_SEED']=='605' and over['client_env']['P3_SNAPSHOT_EVERY']=='3'
    try:
        resolve(a.smoke_run,names[0],dict(P3_SNAPSHOT_P="nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid probability accepted")
    from . import run_gtp_v2, worker_v2
    subset = a.out/"stock_subset.py"
    subset.write_text('from exp.gate_threshold_pareto import run_gtp\nrun_gtp.subset_seen = True\nrun_gtp.main()\n')
    dispatch = []
    for args, env in [([],{}), ([],dict(OSCL_EPISODES="0,1")),([],dict(OSCL_MANIFEST="fixture")),
                      (["--manifest", "fixture"],{}),(["--manifest=fixture"],{})]:
        pkg, driver = ModuleType('exp.gate_threshold_pareto'), ModuleType('exp.gate_threshold_pareto.run_gtp')
        pkg.run_gtp = driver
        driver.WorkerSpec = lambda **kw: kw
        driver.main = lambda: dispatch.append(driver.WorkerSpec(env={"MUJOCO_EGL_DEVICE_ID":"0"}))
        base = dict(P3_SUBSET_SCRIPT=str(subset), P3_SNAPSHOT_DIR=str(a.out/'telemetry'), P3_ENV_SEED="605")
        with patch.dict(sys.modules,{'exp.gate_threshold_pareto':pkg,'exp.gate_threshold_pareto.run_gtp':driver}), patch.dict(os.environ,{**base,**env},clear=True), patch.object(sys,'argv',['test',*args]):
            run_gtp_v2.main()
            assert getattr(driver,'subset_seen',False)==bool(args or env)
        assert dispatch[-1]['env']['MUJOCO_EGL_DEVICE_ID']=='0'
        assert dispatch[-1]['env']['P3_ENV_SEED']=='605'
        assert dispatch[-1]['worker_module'].endswith('worker_v2')
    runner_module, entry = ModuleType('examples.libero.episode_runner'), ModuleType('examples.libero.worker_entry')
    class Runner:
        def __init__(self,*args,**kw):
            self.kw = kw
    runner_module.LiberoEpisodeRunner = Runner
    runner_module.default_client_factory = lambda endpoint: NS()
    observed=[]
    entry.main=lambda: observed.append(runner_module.LiberoEpisodeRunner(None,None))
    pkg=ModuleType('examples.libero'); pkg.episode_runner=runner_module; pkg.worker_entry=entry
    with patch.dict(sys.modules,{'examples.libero':pkg}), patch.dict(os.environ,dict(P3_SNAPSHOT_DIR=str(a.out/'worker'),P3_ENV_SEED='605')), patch.object(sys,'argv',['worker']):
        worker_v2.main()
        assert sys.argv[-2:]==['--seed','605']
        assert observed[0].kw['client_factory'](None).directory==a.out/'worker'
    assert list(zip_compat([1,2],[3,4],strict=True))==[(1,3),(2,4)]
    assert list(zip_compat([],[],strict=True))==[]
    assert list(zip_compat([1],[2,3]))==[(1,2)]
    for left,right in [([1],[2,3]),([1,2],[3])]:
        try:
            list(zip_compat(left,right,strict=True))
        except ValueError:
            pass
        else:
            raise AssertionError('strict mismatch accepted')
    source=a.out/'annotation.py'; source.write_text('value: list[NotDefined] = []\n')
    loader=DeferredSourceLoader('annotation_fixture',str(source))
    module=importlib.util.module_from_spec(importlib.util.spec_from_loader('annotation_fixture',loader))
    loader.exec_module(module)
    assert module.__annotations__['value']=='list[NotDefined]'
    archive=a.out/'telemetry.tar'
    with tarfile.open(archive,'w') as tf:
        data=b'{"ev":"attempt_start"}\n'
        info=tarfile.TarInfo('p3_telemetry/uid_a1/controls.jsonl'); info.size=len(data)
        tf.addfile(info,io.BytesIO(data))
    assert unpack(archive,a.out/'collected')==1
    bad=a.out/'bad.tar'
    with tarfile.open(bad,'w') as tf:
        info=tarfile.TarInfo('p3_telemetry/../../escape'); info.size=1
        tf.addfile(info,io.BytesIO(b'x'))
    try:
        unpack(bad,a.out/'rejected')
    except ValueError:
        pass
    else:
        raise AssertionError('unsafe archive accepted')
    report=dict(PASS=True,payload_files=len(manifest['files']),smoke_arms=len(names),dispatch_paths=len(dispatch),
                isolated_import_modules=len(MODULES),worker_factory_and_seed=True,stock_preserved=True,
                hash_corruption_rejected=True,strict_zip_cases=5,deferred_annotation=True,
                telemetry_unpack=True,unsafe_archive_rejected=True,remote_calls=0,simulator_runs=0)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
