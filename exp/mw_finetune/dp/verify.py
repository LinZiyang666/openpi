"""Reproduce strict tensor loads, identified language statistics and stateless CPU outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from safetensors.torch import load_file
import torch

from .policy import load_policy
from .preprocess import LanguageEncoder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/data/mw_finetune/dp"))
    ap.add_argument("--report", type=Path, default=Path("exp/mw_finetune/dp/reports/verification.json"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); torch.set_num_threads(4)
    tasks = a.root / "mw_dataset_tasks.jsonl"
    if not tasks.exists():
        urllib.request.urlretrieve("https://huggingface.co/datasets/brandonyang/metaworld_ml45/resolve/7c3ffbc972fc6b1b0f701882841c2d17009b71b8/meta/tasks.jsonl", tasks)
    prompts = [json.loads(line)["task"] for line in tasks.read_text().splitlines()]
    emb = LanguageEncoder(str(a.root / "clip_text"))(prompts)
    weights = load_file(str(a.root / "checkpoints/metaworld/model.safetensors"))
    report = {"language": {"model": "openai/clip-vit-large-patch14", "kind": "projected", "tasks": len(prompts)}}
    for stat in ("min", "max"):
        observed = getattr(emb, stat)(dim=0).values
        expected = weights[f"_policy.normalizer.params_dict.lang_emb.input_stats.{stat}"]
        delta = (expected - observed).abs()
        report["language"][stat] = {"mae": delta.mean().item(), "max_abs": delta.max().item()}
        if delta.max() > 1e-4:
            raise ValueError("Language convention no longer matches checkpoint stats")
    for domain in ("libero", "metaworld"):
        p = load_policy(a.root / "checkpoints" / domain, a.device, domain=domain, text_encoder=str(a.root / "clip_text"))
        obs = {"prompt": prompts[0], "seed": 7}
        image = np.zeros((224, 224, 3), np.uint8); state = np.asarray(p.stats["state"]["mean"], np.float32)
        if domain == "libero":
            obs.update({"observation/image": image, "observation/wrist_image": image, "observation/state": state})
        else:
            obs.update(image=image, state=state)
        started = time.monotonic(); first = p.infer(obs)["actions"]
        changed = dict(obs, prompt=prompts[1]); middle = p.infer(changed)["actions"]
        p.reset(); last = p.infer(obs)["actions"]
        assert np.array_equal(first, last) and not np.array_equal(first, middle)
        assert np.isfinite(first).all()
        report[domain] = {"strict_load": p.strict_load, "shape": list(first.shape), "finite": True,
            "stateless_same_seed": True, "language_changes_output": True, "three_calls_s": time.monotonic() - started,
            "checkpoint_sha256": hashlib.sha256((a.root / "checkpoints" / domain / "model.safetensors").read_bytes()).hexdigest()}
        print(json.dumps({domain: report[domain]}), flush=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["language"]), flush=True)


if __name__ == "__main__":
    main()
