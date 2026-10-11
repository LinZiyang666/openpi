"""Pinned weights and compact CLIP text-only export, all inside E's data root."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/data/mw_finetune/dp"))
    a = ap.parse_args(); a.root.mkdir(parents=True, exist_ok=True)
    for domain, rev, sub in [
        ("libero", "3c48ffaab98faacb119beaf2ee12f943503e6492", "dp_libero_lang_v1/30000"),
        ("metaworld", "46c26c0630024f7645f19398cc5be105870ca96f", "dp_metaworld_lang_v1/85000"),
        ("metaworld30k", "46c26c0630024f7645f19398cc5be105870ca96f", "dp_metaworld_lang_v1/30000")]:
        out = a.root / "checkpoints" / domain; out.mkdir(parents=True, exist_ok=True)
        repo = f"brandonyang/dp-{'libero' if domain == 'libero' else 'metaworld'}-checkpoints"
        if not (out / "model.safetensors").exists():
            asset = "physical-intelligence/libero" if domain == "libero" else "brandonyang/metaworld_ml45"
            files = [f"{sub}/model.safetensors", f"{sub}/metadata.pt", f"{sub}/assets/{asset}/norm_stats.json"]
            src = Path(snapshot_download(repo, revision=rev, allow_patterns=files,
                local_dir=a.root / "downloads" / domain))
            for name in files:
                shutil.copy2(src / name, out / Path(name).name)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
                  if p.name in ("metadata.pt", "model.safetensors", "norm_stats.json")}
        (out / "source.json").write_text(json.dumps(dict(repo=repo, revision=rev, subdir=sub, sha256=hashes), indent=2))
    clip = a.root / "clip"
    text = a.root / "clip_text"
    if not (clip / "model.safetensors").exists() and not (text / "model.safetensors").exists():
        snapshot_download("openai/clip-vit-large-patch14", revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
            allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                            "vocab.json", "merges.txt", "special_tokens_map.json"], local_dir=clip)
    if not (text / "model.safetensors").exists():
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        model = CLIPTextModelWithProjection.from_pretrained(clip, local_files_only=True)
        model.save_pretrained(text, safe_serialization=True)
        AutoTokenizer.from_pretrained(clip, local_files_only=True).save_pretrained(text)
    print(json.dumps({"root": str(a.root), "text_bytes": (text / "model.safetensors").stat().st_size}))


if __name__ == "__main__":
    main()
