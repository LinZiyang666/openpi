"""Single-writer episode commits. No scheduler lock or transport thread does I/O."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import threading
import time


class GroupCommit:
    def __init__(self, directory, *, interval=1., max_events=64, max_pending=16384, fsync=os.fsync,
                 extra_streams=()):
        if interval<=0 or max_events<1 or max_pending<max_events:
            raise ValueError('invalid group commit limits')
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.interval,self.max_events,self.fsync=interval,max_events,fsync
        self.queue=queue.Queue(maxsize=max_pending)
        self.failure=None
        self.stats=dict(groups=0,events=0,fsyncs=0,accepted_terminal=0,max_pending=0)
        self.closed=False
        if tuple(extra_streams) not in ((),('fleet_dispatch.jsonl',)):
            raise ValueError('unsupported extra evidence streams')
        self.streams=('dispatch.jsonl','worker_dispatch.jsonl','per_step.jsonl','attempts.jsonl','journal.jsonl',*extra_streams)
        self.thread=threading.Thread(target=self._run,name='r17-evidence-commit',daemon=True)
        self.thread.start()

    def submit(self, rows, callback=None):
        if self.closed or self.failure:
            pass
        try:self.queue.put_nowait((rows,callback))
        except queue.Full as exc:
            self.failure=RuntimeError('bounded evidence queue exhausted; refuse unrecorded completion')
            raise self.failure from exc
        self.stats['max_pending']=max(self.stats['max_pending'],self.queue.qsize())

    def _run(self):
        handles={}
        try:
            # Persist creation before making any terminal episode durable.
            for name in self.streams:
                handles[name]=(self.directory/name).open('a',encoding='utf-8')
            # The fresh driver directory and campaign name must survive too,
            # not just the evidence file entries inside that directory.
            for path in dict.fromkeys((self.directory,self.directory.parent,self.directory.parent.parent)):
                fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
                try:self.fsync(fd)
                finally:os.close(fd)
                self.stats['fsyncs']+=1
            stopping=False
            while not stopping:
                first=self.queue.get()
                if first is None:break
                group=[first]
                until=time.monotonic()+self.interval
                while len(group)<self.max_events:
                    try:event=self.queue.get(timeout=max(0,until-time.monotonic()))
                    except queue.Empty:break
                    if event is None:
                        stopping=True;break
                    group.append(event)
                    if time.monotonic()>=until:break
                # Evidence prerequisites first. A surviving journal line can
                # therefore NEVER refer to lost attempts or per-step evidence.
                dirty=set()
                for rows,_ in group:
                    for name,values in rows.items():
                        if name=='journal.jsonl':continue
                        if name not in handles:pass
                        if values:
                            handles[name].writelines(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in values)
                            dirty.add(name)
                for name in sorted(dirty):
                    handles[name].flush();self.fsync(handles[name].fileno());self.stats['fsyncs']+=1
                terminal=[row for rows,_ in group for row in rows.get('journal.jsonl',())]
                if terminal:
                    journal=handles['journal.jsonl']
                    journal.writelines(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in terminal)
                    journal.flush();self.fsync(journal.fileno());self.stats['fsyncs']+=1
                # Advance completion/retry counters only AFTER both barriers.
                for _,callback in group:
                    if callback is not None:callback()
                self.stats['groups']+=1;self.stats['events']+=len(group)
                self.stats['accepted_terminal']+=sum(row.get('accepted') is True for row in terminal)
        except BaseException as exc:
            self.failure=exc
        finally:
            for stream in handles.values():stream.close()

    def close(self,timeout=120):
        self.closed=True
        # No caller may submit after close; queued events drain before sentinel.
        if self.thread.is_alive():
            self.queue.put(None,timeout=min(timeout,1))
        self.thread.join(timeout)
        if self.thread.is_alive():raise TimeoutError('evidence commit drain exceeded deadline')
        if self.failure:pass
        return dict(self.stats,closed=True,interval_seconds=self.interval,max_group_events=self.max_events)
