"""Replay recorded LIBERO-10 episodes against a serving server as N simulated robots (exp/cluster_sim/DESIGN.md).

The library is the directory written by `record.py split`: one `ep_<task>_<init>.msgpack` per episode holding the
client frames exactly as the worker sent them (select_bundle / episode_start / infer ... / episode_end).
A seeded shuffle orders the 500 episodes; N slots each take the next episode until none are left. Slot i starts
at i * interval / N. After every infer reply a robot waits `interval` seconds (the simulated time to execute the
returned actions plus the network round trip) before sending its next infer; control frames go out at once.
--stride 2 simulates long execution segments: every other infer frame is skipped and the robot waits
2 x 250 ms + 2 ms per reply (half the inference requests).

  python -m exp.cluster_sim.replay --library DIR --url ws://127.0.0.1:8000 --robots 20 --seed 0 --out DIR [--episodes K]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import random
import time

import msgpack
import websockets

CONTROL_PERIOD_S = 0.05  # LIBERO control_freq 20 Hz (measured in libero_sim, libero_10 task 0)
REPLAN_STEPS = 5  # actions a R16 worker executes per infer reply
RTT_S = 0.002
INTERVAL_S = REPLAN_STEPS * CONTROL_PERIOD_S + RTT_S


def load(library):
    episodes = []
    for path in sorted(Path(library).glob('ep_*.msgpack')):
        doc = msgpack.unpackb(path.read_bytes(), raw=False)
        episodes.append(dict(name=path.stem, frames=doc['frames'], kinds=doc['kinds']))
    return episodes


async def robot(slot, url, queue, interval, start, rows):
    await asyncio.sleep(start)
    async with websockets.connect(url, compression=None, max_size=None, open_timeout=None, ping_interval=None) as ws:
        await ws.recv()  # server metadata
        while queue:
            ep = queue.pop(0)
            t0 = time.time()
            infers = 0
            for i, (frame, kind) in enumerate(zip(ep['frames'], ep['kinds'])):
                await ws.send(frame)
                reply = await ws.recv()
                if isinstance(reply, str):
                    raise RuntimeError(f'{ep["name"]} frame {i}: {reply[:2000]}')
                if kind == 'infer':
                    infers += 1
                    if i + 1 < len(ep['kinds']) and ep['kinds'][i + 1] == 'infer':
                        await asyncio.sleep(interval)
            rows.append(dict(slot=slot, episode=ep['name'], start=t0, end=time.time(), infers=infers))


def stride_frames(ep, stride):
    """Long execution segments, simulated: keep every stride-th infer frame of the episode (the 1st, then 1+stride, ...)
    and all control frames; the robot executes stride x REPLAN_STEPS actions per reply."""
    kept, n = [], 0
    for frame, kind in zip(ep['frames'], ep['kinds']):
        if kind == 'infer':
            n += 1
            if (n - 1) % stride:
                continue
        kept.append((frame, kind))
    return dict(ep, frames=[f for f, _ in kept], kinds=[k for _, k in kept])


def interval_for(stride):
    return stride * REPLAN_STEPS * CONTROL_PERIOD_S + RTT_S


async def run(args):
    episodes = load(args.library)
    stride = getattr(args, 'stride', 1)
    if stride > 1:
        episodes = [stride_frames(ep, stride) for ep in episodes]
    random.Random(args.seed).shuffle(episodes)
    queue = episodes[:args.episodes] if args.episodes else list(episodes)
    rows = []
    t0 = time.time()
    await asyncio.gather(*(robot(i, args.url, queue, args.interval, i * args.interval / args.robots, rows)
                           for i in range(args.robots)))
    t1 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'n{args.robots:03d}_s{args.seed}'
    with (out / f'{tag}_episodes.jsonl').open('w') as f:
        for row in sorted(rows, key=lambda r: r['start']):
            f.write(json.dumps(row) + '\n')
    summary = dict(robots=args.robots, seed=args.seed, interval_s=args.interval, stride=stride, url=args.url, library=str(args.library),
                   episodes=len(rows), infers=sum(r['infers'] for r in rows), start=t0, end=t1, total_s=t1 - t0)
    (out / f'{tag}_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True)
    parser.add_argument('--url', default='ws://127.0.0.1:8000')
    parser.add_argument('--robots', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1, help='send every stride-th infer frame (long execution segments)')
    parser.add_argument('--interval', type=float, default=None, help='default: stride x 5 x 50 ms + 2 ms')
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes', type=int, default=0, help='first K of the shuffled library (tuning); 0 = all')
    args = parser.parse_args()
    if args.interval is None:
        args.interval = interval_for(args.stride)
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
