"""Cooperative, single-reader /data budget for this withdrawal/rebuild only."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time
import threading
from unittest.mock import patch

from .audit import io_ready

LOCK=Path('/tmp/r17_longseg_data_reader.lock')
_state=threading.local()

@contextmanager
def filesystem_budget(data_prefix='/data/'):
    """Budget Python archive/hash I/O during a new coordinator prepare only."""
    import builtins,io
    native_builtin,native_io=builtins.open,io.open
    class File:
        def __init__(self,raw):
            self.raw=raw;self.write_bytes=0;self.write_start=time.monotonic()
            self.line_bytes=0;self.line_start=time.monotonic()
        def __getattr__(self,name):return getattr(self.raw,name)
        def __enter__(self):self.raw.__enter__();return self
        def __exit__(self,*a):return self.raw.__exit__(*a)
        def __iter__(self):return self
        def __next__(self):
            line=self.readline()
            if not line:raise StopIteration
            return line
        def readline(self,*a):
            line=self.raw.readline(*a);self.line_bytes+=len(line)
            if self.line_bytes>=256*1024:
                io_ready();time.sleep(max(0,self.line_bytes/2**22-(time.monotonic()-self.line_start)))
                self.line_bytes=0;self.line_start=time.monotonic()
            return line
        def read(self,n=-1):
            blocks=[];remaining=n
            while remaining!=0:
                io_ready();begin=time.monotonic()
                block=self.raw.read(256*1024 if remaining<0 else min(remaining,256*1024))
                if not block:break
                blocks.append(block)
                time.sleep(max(0,len(block)/2**22-(time.monotonic()-begin)))
                if remaining>0:remaining-=len(block)
            return (b'' if 'b' in self.raw.mode else '').join(blocks)
        def readinto(self,buffer):
            block=self.read(len(buffer));buffer[:len(block)]=block;return len(block)
        def write(self,data):
            for i in range(0,len(data),256*1024):
                piece=data[i:i+256*1024];self.raw.write(piece);self.write_bytes+=len(piece)
                if self.write_bytes>=256*1024:
                    io_ready();time.sleep(max(0,self.write_bytes/2**22-(time.monotonic()-self.write_start)))
                    self.write_bytes=0;self.write_start=time.monotonic()
            return len(data)
    def wrapped(native):
        def opened(path,*a,**kw):
            raw=native(path,*a,**kw)
            try:is_data=os.fspath(path).startswith(data_prefix)
            except (TypeError,AttributeError):is_data=False
            return File(raw) if is_data else raw
        return opened
    with reader(),patch.object(builtins,'open',wrapped(native_builtin)),patch.object(io,'open',wrapped(native_io)):
        yield

@contextmanager
def reader():
    if getattr(_state,'depth',0):
        _state.depth+=1
        try:yield
        finally:_state.depth-=1
        return
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        io_ready()
        _state.depth=1
        try:yield
        finally:_state.depth=0

def copy(source,dest):
    source,dest=Path(source),Path(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with reader(),source.open('rb') as src,dest.open('xb') as dst:
        while True:
            io_ready();begin=time.monotonic();block=src.read(256*1024)
            if not block:break
            dst.write(block)
            time.sleep(max(0,len(block)/(4*1024*1024)-(time.monotonic()-begin)))

def dump(path,obj,*,compact=False):
    import json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Metadata normally fits in one block. Large episode plans use bounded
    # writes too, so writeback does not compete with the live journal writer.
    encoder=json.JSONEncoder(indent=None if compact else 2,ensure_ascii=False,
                             separators=(',',':') if compact else None)
    with reader(),path.open('w') as out:
        buffer=[];size=0
        for piece in encoder.iterencode(obj):
            buffer.append(piece);size+=len(piece)
            if size>=256*1024:
                io_ready();begin=time.monotonic();out.write(''.join(buffer))
                time.sleep(max(0,size/(4*1024*1024)-(time.monotonic()-begin)))
                buffer=[];size=0
        out.write(''.join(buffer)+'\n')

def tree(source,dest):
    """Link immutable numeric inputs; copy mutable metadata; never edit links."""
    source,dest=Path(source),Path(dest);dest.mkdir(exist_ok=False)
    for src in source.iterdir():
        if str(source).startswith('/data/'):
            io_ready();time.sleep(.05)
        dst=dest/src.name
        if src.name in ('.eval.lock','TRANSFER_SHA256.json','__pycache__'):continue
        if src.is_symlink():dst.symlink_to(src.resolve())
        elif src.is_dir():tree(src,dst)
        elif src.name in ('CACHE_STUDY.json','manifest.json','ffk_bootstrap.py','ffk_evidence.py'):
            copy(src,dst)
        else:os.link(src,dst)  # immutable descriptors, YAMLs, views and fitted arrays
