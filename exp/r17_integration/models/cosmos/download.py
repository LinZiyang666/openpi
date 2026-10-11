"""Pinned official release download and streaming LFS SHA256 verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ID = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
REVISION = "cb689ec0e3347c13667d70a78a3447388f5c3bb8"
WEIGHT_FILE = "Cosmos-Policy-LIBERO-Predict2-2B.pt"
WEIGHT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"
EMBEDDINGS_SHA256 = "8a03499676c6c196127c577144b9fd09bb02c30f9caf058adbcef09bb99ad8f5"
FILES = (WEIGHT_FILE, "config.json", "libero_dataset_statistics.json", "libero_t5_embeddings.pkl", "README.md")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("/data/mw_finetune/models/cosmos_libero"))
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    if not a.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO_ID, revision=REVISION, allow_patterns=list(FILES), local_dir=str(a.output))
    hashes = {name: sha256_file(a.output / name) for name in FILES}
    if hashes[WEIGHT_FILE] != WEIGHT_SHA256 or hashes["libero_t5_embeddings.pkl"] != EMBEDDINGS_SHA256:
        pass
    report = {"repo_id": REPO_ID, "revision": REVISION, "files": hashes, "verified": True}
    (a.output / "download_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
