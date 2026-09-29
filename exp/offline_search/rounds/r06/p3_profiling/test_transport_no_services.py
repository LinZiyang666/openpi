"""Reuse transport assertions with in-process delivery; never bind a socket.

Networking/SIGKILL tests are deliberately not repeated during this live-pilot
repair: the task prohibits starting or stopping any receiver, including tests.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from . import test_stream as t, test_stream_reader as reader
from .stream_receiver import Store
from .stream_sink import StreamSink


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reader-fixture',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    active={}; serial=[0]
    def receiver(run,**kw):
        serial[0]+=1;address='127.0.0.1:'+str(30000+serial[0])
        handle=SimpleNamespace(store=Store(run),address=address,**dict(delay=kw.get('delay',0),drop_once=kw.get('drop_once',False)))
        active[address]=handle;return handle,None,address
    def stop(handle,thread):active.pop(handle.address)
    def request(address,header,body=b''):
        if header.get('token')!=t.TOKEN:raise RuntimeError('bad token')
        try:return active[address].store.apply(header,body)
        except ValueError as exc:raise RuntimeError(str(exc)) from exc
    def send(self,event):
        if self.address not in active:
            self._memory_connected=False;raise ConnectionRefusedError('test delivery unavailable')
        if not getattr(self,'_memory_connected',False):
            self.stats['reconnects']+=1;self._memory_connected=True
        handle=active[self.address]
        if handle.delay:time.sleep(handle.delay)
        response=handle.store.apply(*event)
        if handle.drop_once and event[0]['op']=='data':
            handle.drop_once=False;self._memory_connected=False;raise EOFError('test lost ACK')
        return response
    with patch.object(t,'receiver',receiver),patch.object(t,'stop',stop),patch.object(t,'request',request),patch.object(StreamSink,'_send',send),patch.object(reader,'receiver',receiver),patch.object(reader,'stop',stop):
        results={name:fn(a.out) for name,fn in [('parity',t.test_parity),('spill',t.test_spill),('outage',t.test_timeout_spill),('close',t.test_close_spill),('multiarm',t.test_multiarm)]}
        with patch.object(sys,'argv',['reader','--fixture',str(a.reader_fixture),'--out',str(a.out/'reader')]):reader.main()
    results.update(PASS=True,sockets_opened=0,services_started=0,network_and_process_restart_retested=False)
    (a.out/'report.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))


if __name__=='__main__':main()
