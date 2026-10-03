"""Run the existing capture validator read-only, with persistent per-arm reports."""
import argparse
import json
from pathlib import Path
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from exp.offline_search.debug import reader, validate

ROOT = Path('/home/weiland/trace_runs/os_closed_loop/r08_main')
OUT = Path('/tmp/r8cb_E5_debug_architecture')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--avoid-front', action='store_true')
    args = parser.parse_args()
    all_specs = json.loads((ROOT / 'arms.json').read_text())
    positions = {s['arm']:i for i,s in enumerate(all_specs)}
    specs = all_specs[args.shard::args.shards]
    if args.reverse:
        specs = specs[::-1]
    if args.limit:
        specs = specs[:args.limit]
    dest = OUT / 'validate'
    dest.mkdir(parents=True, exist_ok=True)
    def one(spec):
        name = spec['arm']
        path = dest / (name + '.json')
        if path.exists():
            return
        if args.avoid_front:
            front = -1
            for log in (OUT/'validate_0.log', OUT/'validate_1.log'):
                for line in log.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get('ev') == 'start':
                        front = max(front, positions[row['arm']])
            if positions[name] <= front+6:
                print(json.dumps(dict(ev='leave_for_original_shards', arm=name, front=front)), flush=True)
                return
        start = time.time()
        print(json.dumps(dict(ev='start', arm=name, time=start)), flush=True)
        arm = reader.open_arm(ROOT, name)
        arm.cache_enabled = False
        try:
            report = validate.validate_arm(arm, expected_pairs=500, require_aug=False)
        except Exception:
            report = dict(arm=name, capture_status='AUDIT_ERROR', error=traceback.format_exc())
        report.update(audit_started=start, audit_finished=time.time(),
                      audit_seconds=time.time() - start, reader_cache_enabled=arm.cache_enabled)
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(ev='done', arm=name, status=report['capture_status'],
            seconds=report['audit_seconds'], decisions=report.get('n_decisions'),
            controls=report.get('n_controls'), missing=len(report.get('capture_missing', [])))), flush=True)
    if args.workers == 1:
        for spec in specs:
            one(spec)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(one, specs))


if __name__ == '__main__':
    main()
