"""Central worker scheduler: batches POST reports, allocate() runs every 5 s, each reply carries that batch's targets.

    python3 -m exp.offline_search.rounds.r17.scheduler.server --port 23500 [--log allocations.jsonl]

Stateless: reports live in memory for TTL_S; allocation changes are appended to the JSONL log for humans.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import traceback

from . import VERSION
from .allocate import live, plan

RECOMPUTE_S = 5
# After a (re)start, answer no targets until every live batch has reported once (client period 15 s);
# a partial picture would hand one batch the whole pool and retire everybody else's workers.
STARTUP_S = 30


class Scheduler:
    def __init__(self, log):
        self.log = Path(log)
        self.lock = threading.Lock()
        self.started = time.time()
        self.reports = {}
        self.state = dict(time=None, ready=False, targets={}, capacity={}, shares={})

    def report(self, body):
        now = time.time()
        with self.lock:
            self.reports[body['batch']] = dict(body, received=now)
            targets = self.state['targets'].get(body['batch']) if self.state['ready'] else None
        return dict(targets=targets, time=now)

    def recompute(self):
        now = time.time()
        # Operator caps on desire ({batch: workers}, re-read every recompute): batches whose frozen client
        # cannot measure throughput (sched-1019 group-commit drivers) and whose server GPU is unreadable.
        # floors.json likewise raises desire when a client's step test misreads (stale node snapshots).
        bounds = {}
        for name in ('caps', 'floors'):
            try:
                bounds[name] = json.loads((self.log.parent / (name + '.json')).read_text())
            except (OSError, ValueError):
                bounds[name] = {}
        def bounded(b, r):
            desired = max(r['desired'], bounds['floors'].get(b, 0))
            return dict(r, desired=min(desired, bounds['caps'].get(b, desired))) if not r['done'] else r
        with self.lock:
            self.reports = live(self.reports, now)
            reports = {b: bounded(b, r) for b, r in self.reports.items()}
            previous = self.state['targets']
        result = plan(reports, now, previous)
        with self.lock:
            changed = result['targets'] != self.state['targets']
            self.state = dict(result, time=now, ready=now - self.started >= STARTUP_S)
        if changed:
            row = dict(time=now, targets=result['targets'], shares=result['shares'],
                       desired={b: r['desired'] for b, r in reports.items()},
                       active={b: sum(r['active'].values()) for b, r in reports.items()},
                       done=sorted(b for b, r in reports.items() if r['done']))
            with self.log.open('a') as out:
                out.write(json.dumps(row, sort_keys=True) + '\n')

    def view(self):
        with self.lock:
            return dict(self.state, version=VERSION, now=time.time(), started=self.started,
                        reports=dict(self.reports))

    def loop(self):
        while True:
            try:
                self.recompute()
            except Exception:
                traceback.print_exc()
            time.sleep(RECOMPUTE_S)


def handler(scheduler):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/healthz':
                self.reply(dict(ok=True, version=VERSION))
            elif self.path == '/v1/state':
                self.reply(scheduler.view())
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != '/v1/report':
                return self.send_error(404)
            self.reply(scheduler.report(json.loads(self.rfile.read(int(self.headers['Content-Length'])))))

        def reply(self, value):
            data = json.dumps(value, sort_keys=True).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=23500)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--log', default=None, help='allocation-change JSONL (default scheduler_<port>.jsonl)')
    args = parser.parse_args()
    scheduler = Scheduler(args.log or 'scheduler_%d.jsonl' % args.port)
    server = ThreadingHTTPServer((args.host, args.port), handler(scheduler))
    threading.Thread(target=scheduler.loop, daemon=True).start()
    print(json.dumps(dict(event='listening', version=VERSION, host=args.host, port=args.port,
                          log=str(scheduler.log), time=scheduler.started)), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
