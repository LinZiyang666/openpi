"""Coordinator plumbing with mocked tether; no remote command is executed."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from . import collect_client, run_gtp_v2
from .client_plan import resolve
from .stream_receiver import digest
from .test_stream import setup, journal, TOKEN


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    run=setup(a.out/'run');(run/'state').mkdir()
    (run/'state/p3_stream.token').write_text(TOKEN)
    (run/'schedule_v2.json').write_text(json.dumps([dict(arm='test',client_env=dict(P3_ENV_SEED='603'))]))
    plan=resolve(run,'test',dict(P3_STREAM_PORT='54321',P3_STREAM_QUEUE_BYTES='1048576'))
    assert plan['client_env']['P3_STREAM']=='ziyanglin.com:54321'
    assert plan['client_env']['P3_STREAM_TOKEN']==TOKEN and plan['client_env']['P3_STREAM_RUN']==run.name
    assert 'P3_STREAM' not in resolve(run,'test',{})['client_env']
    pkg,driver=ModuleType('exp.gate_threshold_pareto'),ModuleType('exp.gate_threshold_pareto.run_gtp')
    pkg.run_gtp=driver;driver.WorkerSpec=lambda **kw:kw;captured=[]
    driver.main=lambda:captured.append(driver.WorkerSpec(env={'MUJOCO_EGL_DEVICE_ID':'0'}))
    with patch.dict(sys.modules,{'exp.gate_threshold_pareto':pkg,'exp.gate_threshold_pareto.run_gtp':driver}), patch.dict(os.environ,plan['client_env'],clear=True), patch.object(sys,'argv',['run_gtp_v2']):
        run_gtp_v2.main()
    assert captured[0]['env']['P3_STREAM']==plan['client_env']['P3_STREAM']
    assert captured[0]['env']['MUJOCO_EGL_DEVICE_ID']=='0'
    assert captured[0]['env']['P3_STREAM_QUEUE_BYTES']=='1048576'
    archive=a.out/'source.tar'
    with tarfile.open(archive,'w') as tf:tf.add(a.reference,arcname='p3_telemetry')
    journal(run,'test',['test:eval:0:0'])
    cleanup=[]
    def fake(cmd,**kw):
        assert cmd[0]=='tether'
        if cmd[1]=='pull':shutil.copyfile(archive,cmd[-1])
        elif 'rm -r --' in cmd[-1]:cleanup.append(cmd[-1])
        else:return SimpleNamespace(stdout=digest(archive)+'  remote.tar\n')
        return SimpleNamespace(stdout='')
    with patch.object(subprocess,'run',side_effect=fake),patch.object(sys,'argv',['collect_client','--run-root',str(run),'--arm','test','--cleanup']):collect_client.main()
    assert len(cleanup)==1 and (run/'runs/test/p3_stream_verified.json').exists()
    def corrupt(cmd,**kw):
        if cmd[1]=='exec' and 'rm -r --' not in cmd[-1]:return SimpleNamespace(stdout='0'*64+'  remote.tar\n')
        return fake(cmd,**kw)
    with patch.object(subprocess,'run',side_effect=corrupt),patch.object(sys,'argv',['collect_client','--run-root',str(run),'--arm','test','--cleanup']):
        try:collect_client.main()
        except ValueError:pass
        else:raise AssertionError('bad remote archive hash accepted')
    assert len(cleanup)==1
    result=dict(PASS=True,remote_calls=0,mocked_cleanup_requests=1,stream_env_forwarding=True,
                default_env_unchanged=True,file_mode_cleanup_verified=True,bad_archive_prevents_cleanup=True)
    (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
