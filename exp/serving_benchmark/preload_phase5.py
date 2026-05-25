"""One-shot helper: preload phase5 G5 warmup buffer + load phase5 eval yaml
as the server's ``default`` bundle so subsequent ``main.py`` clients (which
do not send ``select_bundle``) automatically use the real phase5 composite
judge path (FULL_HIT / WARM_START / MISS mix instead of always_hit).

Usage::

    python -m exp.serving_benchmark.preload_phase5 \\
        --host 149.165.151.106 --port 8000 \\
        --cell-id phase5_g5_g6__fh0.3_ws0.3 \\
        --eval-yaml exp/verdict_factor_judge/data/phase5_systematic/eval_yaml/\\
spatial16_w8_d4_phase5_g5_g6__fh0.3_ws0.3.yaml

The eval yaml must already exist on disk (emit it via ``phase5.runner --mode
emit-eval-yamls`` first if missing). For G5 cells the warmup raw is sourced
from phase3 g6 / phase4 p1+p2 historical dumps — those must also already
exist on disk; this script does not regenerate them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from openpi_client.websocket_client_policy import WebsocketClientPolicy

from exp.verdict_factor_judge.phase3.threshold_solver import load_per_key_finite_history
from exp.verdict_factor_judge.phase5.spec import generate_all_cells
from exp.verdict_factor_judge.phase5.spec import warmup_raw_path_for_cell


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--cell-id", required=True,
                   help="substring of cell.yaml_id (e.g. phase5_g5_g6__fh0.3_ws0.3)")
    p.add_argument("--eval-yaml", required=True,
                   help="path to eval yaml on disk")
    p.add_argument("--cfg-id", default="spatial16_w8_d4")
    p.add_argument("--phase3-warmup-dir", default="")
    p.add_argument("--phase4-warmup-raw-dir", default="")
    p.add_argument("--phase5-warmup-dir", default="")
    p.add_argument("--bundle-id", default="default",
                   help="server-side bundle id to load yaml under (default: 'default')")
    args = p.parse_args()

    eval_yaml_path = Path(args.eval_yaml)
    if not eval_yaml_path.exists():
        print(f"ERR: --eval-yaml {eval_yaml_path} does not exist", file=sys.stderr)
        return 2

    cells = generate_all_cells(cfg_id=args.cfg_id)
    matches = [c for c in cells if args.cell_id in c.yaml_id]
    if not matches:
        print(f"ERR: no cell matches --cell-id {args.cell_id!r}", file=sys.stderr)
        return 2
    cell = matches[0]
    print(f"[preload] resolved cell.yaml_id={cell.yaml_id}", flush=True)

    raw_path = warmup_raw_path_for_cell(
        cell,
        phase5_warmup_dir=Path(args.phase5_warmup_dir) if args.phase5_warmup_dir
        else Path("exp/verdict_factor_judge/data/phase5_systematic/warmup_factor_raw"),
        phase3_warmup_dir=Path(args.phase3_warmup_dir) if args.phase3_warmup_dir
        else Path("exp/verdict_factor_judge/data/phase3/warmup_factor_raw"),
        phase4_warmup_raw_dir=Path(args.phase4_warmup_raw_dir) if args.phase4_warmup_raw_dir
        else Path("exp/verdict_factor_judge/data/phase4/warmup_factor_raw"),
    )
    if not raw_path.exists():
        print(f"ERR: warmup raw jsonl missing at {raw_path}", file=sys.stderr)
        return 2

    declared = list(cell.declared_keys)
    print(f"[preload] loading warmup buffer from {raw_path} keys={declared}", flush=True)
    buffer = load_per_key_finite_history(raw_path, declared)
    print(f"[preload] buffer has {len(buffer)} keys", flush=True)

    with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
        ack1 = ctl.preload_normalizer_buffer(args.bundle_id, buffer)
        print(f"[preload] preload_normalizer_buffer ack={ack1}", flush=True)
        yaml_content = eval_yaml_path.read_text(encoding="utf-8")
        ack2 = ctl.load_cache_config(
            yaml_content=yaml_content,
            yaml_id=args.bundle_id,
        )
        print(f"[preload] load_cache_config ack={ack2}", flush=True)
    print("[preload] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
