"""CPU-only memfd copy/RPC cost on recorded B pixels and bf16 tokens.

Uses a socketpair and an owned CPU echo process, with no listeners or weights.
This isolates transport and host copies; CUDA transfers are not measured.
"""
import argparse
import json
import multiprocessing
from pathlib import Path
import socket
import time


def serve(sock):
    from openpi.serving.split_service import StageService
    class Echo(StageService):
        def dispatch(self, operation, args, *rest):
            if operation == "echo":
                return args, False
            return super().dispatch(operation, args, *rest)
    Echo(None).serve_channel(sock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi05", "groot"), required=True)
    parser.add_argument("--calls", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from exp.decouple.recorded import load
    from openpi.serving.split_transport import Channel
    data = load(args.model, limit=1)
    payload = dict(images=[np.array(data.token[name][0], np.float32) for name in ("img0", "img1")],
                   tokens=torch.cat([torch.from_numpy(np.array(data.token[name][0])) for name in ("v0", "v1")]).bfloat16())
    left, right = socket.socketpair()
    child = multiprocessing.get_context("spawn").Process(target=serve, args=(right,))
    child.start()
    right.close()
    client = Channel(left)
    try:
        for _ in range(3):
            restored = client.call("echo", payload)
        assert torch.equal(restored["tokens"].view(torch.uint8), payload["tokens"].view(torch.uint8))
        latencies = []
        start_cpu = time.process_time()
        for _ in range(args.calls):
            start = time.perf_counter_ns()
            client.call("echo", payload)
            latencies.append((time.perf_counter_ns()-start)/1e6)
        cpu_ms = 1000*(time.process_time()-start_cpu)/args.calls
    finally:
        client.close()
        child.join(10)
        if child.is_alive():
            child.terminate()
            child.join(10)
    if child.exitcode:
        raise RuntimeError(f"owned echo process exited {child.exitcode}")
    result = dict(model=args.model, calls=args.calls, scope="CPU echo; input/output copies, no CUDA transfer",
                  bytes_per_direction=sum(a.nbytes for a in payload["images"])+payload["tokens"].numel()*2,
                  front_cpu_ms_per_call=cpu_ms, latency_mean_ms=float(np.mean(latencies)),
                  latency_p50_ms=float(np.percentile(latencies,50)), latency_p99_ms=float(np.percentile(latencies,99)))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
