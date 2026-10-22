"""Record every client frame a resident receives, then cut the streams into one replay file per episode.

Raw log: one msgpack stream per server connection, `<raw>/c<pid>_<n>.msgpack`, each item {'t': unix time, 'f': frame}
with the frame exactly as received. Only the client side is kept (owner 10-21: no model-side data).

  python -m exp.cluster_sim.record split --raw RAW --out LIBRARY

writes `<LIBRARY>/ep_<task>_<init>.msgpack` = {'task', 'init', 'success', 'frames', 'kinds', 't'} for the last complete
occurrence (episode_start ... episode_end) of every (task_id, orig_init_state_idx); the connection's latest
select_bundle frame is prepended so a replayed episode binds the same arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import msgpack


def kind(frame):
    """'select_bundle' / 'episode_start' / 'episode_end' / other control names, or 'infer'."""
    head = msgpack.unpackb(frame, raw=False, strict_map_key=False)
    return head.get('__ctrl__', 'infer') if isinstance(head, dict) else 'infer'


def split(raw, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    episodes = {}
    for path in sorted(Path(raw).glob('c*.msgpack')):
        bundle, current = None, None
        with path.open('rb') as f:
            for item in msgpack.Unpacker(f, raw=False, max_buffer_size=1 << 31):
                frame, k = item['f'], kind(item['f'])
                if k == 'select_bundle':
                    bundle = (item['t'], frame)
                    continue
                if k == 'episode_start':
                    start = msgpack.unpackb(frame, raw=False, strict_map_key=False)
                    extra = start.get('__extra__') or {}
                    current = dict(task=int(extra['task_id']), init=int(extra['orig_init_state_idx']),
                                   frames=[] if bundle is None else [bundle[1]],
                                   kinds=[] if bundle is None else ['select_bundle'],
                                   t=[] if bundle is None else [bundle[0]])
                if current is None:
                    continue
                current['frames'].append(frame)
                current['kinds'].append(k)
                current['t'].append(item['t'])
                if k == 'episode_end':
                    current['success'] = bool(msgpack.unpackb(frame, raw=False).get('__success__'))
                    key = current['task'], current['init']
                    if key not in episodes or episodes[key]['t'][0] < current['t'][0]:
                        episodes[key] = current
                    current = None
    for (task, init), ep in episodes.items():
        (out / f'ep_{task}_{init:02d}.msgpack').write_bytes(msgpack.packb(ep, use_bin_type=True))
    summary = dict(episodes=len(episodes), infers=sum(e['kinds'].count('infer') for e in episodes.values()),
                   successes=sum(e['success'] for e in episodes.values()),
                   per_task={t: sum(1 for k in episodes if k[0] == t) for t in sorted({k[0] for k in episodes})})
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    s = sub.add_parser('split')
    s.add_argument('--raw', required=True)
    s.add_argument('--out', required=True)
    args = parser.parse_args()
    split(args.raw, args.out)


if __name__ == '__main__':
    main()
