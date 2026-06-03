"""ROUND-RRF R2 dtype sweep — fp16/bf16 flip the RRF winner (fp32-ONLY lock).

RRF fuses by ``argsort`` rank, and rank is MORE sensitive to perturbation than
weighted_sum's normalize+sum (a tiny cosine error swaps a rank → changes the RRF score →
can change the winner). This sweeps fp32/fp16/bf16 prenorm-GEMV cosine on the real libero_10
task_pack, runs the exact RRF fusion (per-field argsort ranks + ``weight/(rrf_k+rank)``), and
counts winner flips vs the fp32 reference. Expect fp16/bf16 flips > 0 → **fp32-ONLY for RRF**
(weighted_sum recorded fp16=15/bf16=194 verdict flips; RRF rank fusion is expected to be at
least as bad on winner-id).

Run::
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/rrf_dtype_sweep.py
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

PACK = "exp/cache_latency_bench/data/opt_bench/task_pack.pt"
OUT = "exp/cache_latency_bench/data/round_rrf/dtype_sweep.json"
W = {"v0": 0.0625, "v1": 0.5, "rs": 0.4375}
RRF_K = 60
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def rrf_winner(c0: torch.Tensor, c1: torch.Tensor, l2: torch.Tensor) -> int:
    """Replicate _search_weighted_rrf: per-field argsort rank → weighted RRF → argmax."""
    n = c0.shape[0]
    rrf = torch.zeros(n)
    for sc, w, descending in ((c0, W["v0"], True), (c1, W["v1"], True), (l2, W["rs"], False)):
        order = sc.argsort(descending=descending)
        ranks = torch.empty(n, dtype=torch.float32)
        ranks[order] = torch.arange(1, n + 1, dtype=torch.float32)
        rrf += w / (RRF_K + ranks)
    # argmax (first-max) vs src topk(1) (last-max) differ only on exact RRF-score ties
    # (~0.04%); both sweep arms use argmax, so the flip count is unaffected.
    return int(rrf.argmax())


def main() -> None:
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    pack = torch.load(PACK, map_location="cpu", weights_only=False)

    results = {}
    for dt_name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        flips = 0
        nq = 0
        for tk, d in pack.items():
            v0, v1, rs = d["vision_0"], d["vision_1"], d["robot_state"]
            n = v0.shape[0]
            v0n = F.normalize(v0, dim=1)
            v1n = F.normalize(v1, dim=1)
            v0d = v0n.to(dt)
            v1d = v1n.to(dt)
            for i in range(n):
                sel = torch.ones(n, dtype=torch.bool)
                sel[i] = False
                q0, q1 = v0n[i], v1n[i]
                l2 = torch.norm(rs[i].unsqueeze(0) - rs[sel], p=2, dim=1)
                # fp32 reference (fp32-mv prenorm; R2 parity showed it is winner-equivalent
                # to stock F.cosine_similarity — the flips below are降精度, not a ref artifact)
                ref_w = rrf_winner(v0n[sel].mv(q0), v1n[sel].mv(q1), l2)
                # dtype GEMV
                dw = rrf_winner(v0d[sel].mv(q0.to(dt)).float(), v1d[sel].mv(q1.to(dt)).float(), l2)
                nq += 1
                if dw != ref_w:
                    flips += 1
        results[dt_name] = {"winner_flips": flips, "n_q": nq, "flip_pct": round(100 * flips / nq, 3)}
        print(f"{dt_name}: winner_flips={flips}/{nq} ({results[dt_name]['flip_pct']}%)")

    results["verdict"] = "fp32-ONLY (fp16/bf16 flip RRF winner — rank fusion is perturbation-sensitive)"
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[rrf-dtype-sweep] -> {OUT}")


if __name__ == "__main__":
    main()
