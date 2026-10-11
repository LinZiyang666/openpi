"""CPU-only bounded two-port NAT admission test, no policy or GPU imports."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host',choices=('local','h100'),required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    ports=[23156,23157] if a.host=='local' else [23256,23257]
    address='ziyanglin.com' if a.host=='local' else '149.165.153.233'
    node='timan107' if a.host=='local' else 'timan108'
    script='''import socket,threading,time,json,os
socks=[]
def accept(s):
 s.settimeout(25)
 try:
  c,_=s.accept()
  with c: c.sendall(b"r17-Q-CPU-route-v1")
 except TimeoutError: pass
for port in PORTS:
 s=socket.socket(); s.bind(("0.0.0.0",port)); s.listen(2); socks.append(s)
print("READY "+str(os.getpid()),flush=True)
threads=[threading.Thread(target=accept,args=(s,)) for s in socks]
for t in threads: t.start()
for t in threads: t.join()
for s in socks: s.close()
'''.replace('PORTS',repr(ports))
    argv=['/home/weiland/projects/openpi/.venv/bin/python','-u','-c',script]
    if a.host=='h100':
        argv=['tether','exec','--safe','--timeout','35s','h100','--','env','CUDA_VISIBLE_DEVICES=',
              '/home/exouser/openpi/.venv/bin/python','-u','-c',script]
    child=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    report=dict(status='FAIL',cpu_only=True,actual_gpu_executed=False,ports=ports,node=node,address=address,owned_pid=child.pid)
    try:
        ready=child.stdout.readline()
        assert ready.startswith('READY '),ready+child.stderr.read()
        report['listener_pid']=int(ready.split()[1])
        probe='import socket,json; rows=[]\nfor p in '+repr(ports)+':\n with socket.create_connection(('+repr(address)+',p),timeout=10) as s: rows.append(s.recv(100).decode())\nassert rows==["r17-Q-CPU-route-v1"]*2\nprint(json.dumps(rows))'
        result=subprocess.run(['tether','exec','--safe','--timeout','25s',node,'--',
            '/scratch/zixuans8/mw_r17/envs/sim/bin/python','-c',probe],capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stdout+result.stderr
        report.update(status='PASS',echoes=json.loads(result.stdout))
        assert child.wait(timeout=30)==0
    except BaseException as exc:
        report['error']=str(exc)
    finally:
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=3)
            except subprocess.TimeoutExpired: child.kill();child.wait(timeout=3)
        report['exit_code']=child.returncode
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
