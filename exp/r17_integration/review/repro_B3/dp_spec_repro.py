"""CPU repro: DP-LIBERO consumption of P2 LIBERO benchmark documents (toy DP, no weights)."""
import sys, json
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src"]
from dataclasses import asdict
import torch
from exp.mw_finetune.dp.model import DiffusionPolicy
from exp.r17_integration.models.dp.staged import DPAdapter
from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec, SUITES
from pathlib import Path
maps = {s: {"suite": s, "tasks": [{"task_id": i, "name": f"task_{i}", "language": f"{s} instruction {i}"} for i in range(10)]} for s in SUITES}
system = LiberoBenchmarkSpec.from_task_maps(maps, wire_size=96)
released = LiberoBenchmarkSpec.from_task_maps(maps, wire_size=256, seed=42, seed_mode="fixed", replan_steps=32,
    max_steps_by_suite=(280, 280, 300, 520), protocol_id="libero_released_reference_r17_v1")
cfg = json.loads(Path("/home/weiland/projects/openpi/exp/r17_integration/models/dp/samples/audited_libero_metadata.json").read_text())["config"]["model"]
cfg.update(lang_emb_dim=8, n_layer=1, n_cond_layers=1, n_head=2, n_emb=16, p_drop_emb=0.0, p_drop_attn=0.0)
model = DiffusionPolicy(cfg).eval()
stats = {n: {"q01": [-1.0]*d, "q99": [1.0]*d} for n, d in (("actions", 7), ("state", 8))}
lang = lambda p: torch.zeros(len(p), 8)
def build(docs):
    return DPAdapter(model, stats, lang, domain="libero", weight_identity="toy", libero_benchmark_specs=docs)
for label, docs in (("P2 emitted manifest (libero_bridge --output)", [json.loads(json.dumps(system.manifest()))]),
                    ("P2 released-reference spec (Fast-WAM-only protocol)", [asdict(released)])):
    try:
        a = build(docs)
        print(f"{label}: ACCEPTED, allowed protocols={len(a.metadata().allowed_benchmark_protocols)}, layout replan={a.metadata().layout.replan_steps}")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")
