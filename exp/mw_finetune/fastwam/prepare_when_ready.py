"""CPU-only preparation while A's shared dataset download completes."""
import argparse
import json
from pathlib import Path
import time

from .prepare_data import prepare


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="/data/mw_finetune/datasets/metaworld_mt50")
    p.add_argument("--destination", default="/data/mw_finetune/fastwam/mt50_224")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout-seconds", type=int, default=3600)
    a = p.parse_args()
    start = time.monotonic()
    while time.monotonic() - start < a.timeout_seconds:
        try:
            prepare(a.source, a.destination, a.workers)
            break
        except ValueError as exc:
            if "incomplete/noncontiguous" not in str(exc) and "Expected 204806 frames" not in str(exc):
                raise
            print(json.dumps({"waiting_for_download": str(exc)}), flush=True)
            time.sleep(30)
    else:
        raise TimeoutError("Dataset download is still incomplete; rerun this resumable command")
