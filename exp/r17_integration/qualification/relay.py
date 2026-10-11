"""Run-scoped CPU tether relay; H100 has no tether installation/credentials."""
from __future__ import annotations
import argparse
import http.server
import json
import os
from pathlib import Path
import secrets
import subprocess
import threading
import time
import urllib.request


class Broker:
    def __init__(self,run,port=23158):
        self.run=Path(run)
        self.token=secrets.token_hex(32)
        self.url=f'http://ziyanglin.com:{port}'
        self.children=[]
        self.lock=threading.Lock()
        broker=self
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                if self.headers.get('Authorization')!='Bearer '+broker.token:
                    self.send_error(403);return
                try:
                    n=int(self.headers.get('Content-Length','0'))
                    if not 0<n<=32768: raise ValueError('oversized request')
                    args=json.loads(self.rfile.read(n))
                    result=broker.execute(args)
                except BaseException as exc:
                    result=dict(rc=1,stdout='',stderr=f'{type(exc).__name__}: {exc}')
                data=json.dumps(result).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers()
                self.wfile.write(data)
        self.server=http.server.ThreadingHTTPServer(('0.0.0.0',port),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def command(self,argv,timeout=570):
        child=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        with self.lock:
            self.children.append(child)
            (self.run/'RELAY_PIDS.json').write_text(json.dumps([dict(pid=c.pid) for c in self.children]))
        try:
            out,err=child.communicate(timeout=timeout)
            if len(out)+len(err)>4*1024**2: raise ValueError('relay output exceeds 4 MiB')
            return dict(rc=child.returncode,stdout=out,stderr=err)
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=2)
                except subprocess.TimeoutExpired: child.kill();child.wait(timeout=2)

    def execute(self,args):
        if not isinstance(args,list) or not args or not all(isinstance(v,str) for v in args):
            raise ValueError('argv required')
        if args[0]=='probe': return dict(rc=0,stdout='Q relay CPU PASS\n',stderr='')
        if args[0]=='exec':
            sep=args.index('--')
            node=args[sep-1]
            command=args[sep+1:]
            if node not in ('timan107','timan108','weilandserver'): raise ValueError('node denied')
            # Only Q-qualified workers, exact Q run recovery or PID cleanup are admissible.
            text=' '.join(command)
            if 'qualification' not in text or any(x in text for x in ('/archive','/data/i018','/dev/shm/i018','/data/idea_sprint','/home/weiland/idea-sprint')):
                raise ValueError('command outside Q scope')
            return self.command(['tether',*args])
        if args[0]=='pull':
            source,destination=args[1:3]
            node,path=source.split(':',1)
            if node not in ('timan107','timan108','weilandserver'):
                raise ValueError('pull outside Q scope')
            d=Path(destination)
            from .archive_stage import RUN_ROOT,stage_identity,file_sha256
            if d.parent!=RUN_ROOT/self.run.name or d.name not in ('workers.tar','libero.tar'):
                raise ValueError('destination denied')
            kind=d.stem
            expected=stage_identity(path,self.run.name,kind)
            local=self.run/'relay'/d.name
            local.parent.mkdir(exist_ok=True)
            result=self.command(['tether','pull',source,str(local),'--timeout','35s'],40)
            if result['rc']: return result
            if local.stat().st_size>512*1024**2: raise ValueError('relay archive exceeds 512 MiB')
            if file_sha256(local)!=expected:raise ValueError('relay pulled archive SHA mismatch')
            # Tether also restricts H100 push destinations. Push into /tmp,
            # then use a bounded CPU copy into this exact Q run with SHA checks.
            incoming=Path(path)
            result=self.command(['tether','exec','--safe','--timeout','10s','h100','--',
                'mkdir','-p',str(incoming.parent)],15)
            if result['rc']:return result
            result=self.command(['tether','push',str(local),'h100:'+str(incoming),'--timeout','35s'],40)
            if result['rc']:return result
            runtime='/data/mw_finetune/qualification/runtime_q1'
            result=self.command(['tether','exec','--safe','--timeout','20s','h100','--','env',
                'CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+runtime,
                '/home/exouser/openpi/.venv/bin/python','-m','exp.r17_integration.qualification.archive_stage',
                '--install',str(incoming),'--destination',str(d),'--run-key',self.run.name,
                '--kind',kind,'--sha256',expected],25)
            if result['rc']:return result
            receipt=json.loads(result['stdout'])
            if receipt['status']!='PASS' or receipt['archive_sha256']!=expected:
                raise ValueError('H100 relay install receipt differs')
            (self.run/('RELAY_TRANSFER_'+kind+'.json')).write_text(json.dumps(dict(source=source,
                local_archive=str(local),local_sha_verified=True,h100=receipt,cpu_only=True))+'\n')
            return result
        raise ValueError('relay operation denied')

    def close(self):
        self.server.shutdown();self.server.server_close()
        with self.lock:
            children=list(self.children)
        for child in children:
            if child.poll() is None: child.terminate()
        until=time.monotonic()+2
        while time.monotonic()<until and any(c.poll() is None for c in children): time.sleep(.1)
        for child in children:
            if child.poll() is None: child.kill()
            child.wait(timeout=2)
        (self.run/'RELAY_CLOSED.json').write_text(json.dumps(dict(status='PASS',pids=[dict(pid=c.pid,exit_code=c.returncode) for c in children])))


def main():
    import sys
    req=urllib.request.Request(os.environ['Q_RELAY_URL'],data=json.dumps(sys.argv[1:]).encode(),
         headers={'Authorization':'Bearer '+os.environ['Q_RELAY_TOKEN'],'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as response: result=json.load(response)
    sys.stdout.write(result['stdout']);sys.stderr.write(result['stderr'])
    return result['rc']


if __name__=='__main__': raise SystemExit(main())
