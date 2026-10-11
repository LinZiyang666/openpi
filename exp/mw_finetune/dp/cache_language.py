"""One canonical, transferable cache of frozen training instruction embeddings."""
import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import save_file
import torch

from .preprocess import LanguageEncoder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("/data/mw_finetune/dp/mt50_96"))
    ap.add_argument("--text-encoder", default="/data/mw_finetune/dp/clip_text")
    a = ap.parse_args(); torch.set_num_threads(4)
    manifest = json.loads((a.dataset / "manifest.json").read_text())
    path = a.dataset / "language.safetensors"
    if not path.exists():
        embeddings = LanguageEncoder(a.text_encoder)([manifest["tasks"][str(i)] for i in range(len(manifest["tasks"]))]).clone()
        save_file({"embeddings": embeddings}, str(path), metadata={"model": "openai/clip-vit-large-patch14",
            "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41", "kind": "projected"})
    print(json.dumps({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
