"""Download the immutable community release, with an optional full weight hash."""
import argparse
import hashlib
import json
from pathlib import Path

from exp.mw_finetune.cosmos.policy import REPO_ID, REVISION, WEIGHT_SHA256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/data/mw_finetune/models/cosmos_metaworld"))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(REPO_ID, revision=REVISION, local_dir=a.root,
                      allow_patterns=["README.md", "dataset_statistics.json", "t5_embeddings.pkl", "consolidated.pt"])
    weight = a.root / "consolidated.pt"
    manifest = {"repo_id": REPO_ID, "revision": REVISION, "weight_size": weight.stat().st_size,
                "expected_weight_sha256": WEIGHT_SHA256}
    if a.verify:
        h = hashlib.sha256()
        with weight.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        manifest["observed_weight_sha256"] = h.hexdigest()
        if h.hexdigest() != WEIGHT_SHA256:
            raise ValueError("checkpoint SHA256 mismatch")
    (a.root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
