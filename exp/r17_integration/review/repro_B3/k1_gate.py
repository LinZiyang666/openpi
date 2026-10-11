import sys
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src"]
import json, torch
from pathlib import Path
from exp.mw_finetune.dp.model import DiffusionPolicy
from exp.r17_integration.models.dp.staged import DPAdapter
from exp.r17_integration.models.dp.provider import register_provider
from exp.r17_integration.models.dp.serve import plain_config
cfg = json.loads(Path("/home/weiland/projects/openpi/exp/r17_integration/models/dp/samples/audited_libero_metadata.json").read_text())["config"]["model"]
cfg.update(action_dim=4, lowdims=[{"key": "state", "dim": 4}], lang_emb_dim=8, n_layer=1, n_cond_layers=1, n_head=2, n_emb=16, p_drop_emb=0.0, p_drop_attn=0.0)
m = DiffusionPolicy(cfg).eval()
a = DPAdapter(m, {n: {"q01": [-1.0]*4, "q99": [1.0]*4} for n in ("actions", "state")}, lambda p: torch.zeros(len(p), 8), domain="metaworld", weight_identity="toy")
p = register_provider(a, provider_id="reviewB3_k1")
def attempt(label, **kw):
    try:
        c = plain_config(p, **kw); print(f"{label}: ACCEPTED k={c.adapter_binding['num_steps']} commit={c.adapter_binding['commit_horizon']}")
    except Exception as e:
        print(f"{label}: REJECTED ({type(e).__name__}: {str(e)[:90]})")
attempt("K1 unlabelled", num_steps=1)
attempt("K1 labelled plain commit5", num_steps=1, allow_lossy_sampler=True, sampler_study_label="study")
attempt("K1 labelled commit10", num_steps=1, commit_horizon=10, allow_lossy_sampler=True, sampler_study_label="study")
attempt("K1 labelled + CP1 retrieval", num_steps=1, retrieval=True, allow_lossy_sampler=True, sampler_study_label="study")
attempt("K2 ordinary", num_steps=2)
attempt("K2 with lossy flag but K2", num_steps=2, allow_lossy_sampler=True, sampler_study_label="study")
