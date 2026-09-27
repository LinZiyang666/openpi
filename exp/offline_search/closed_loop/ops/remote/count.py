"""count.py JOURNAL [--manifest PATH] [--arm NAME] -> complete success rows.

--manifest-info PATH prints distinct-pair EXPECT and selection SHA256. This file
and run_gtp_subset.py are standalone stdlib tools on the remote island.
"""
import argparse
import json
from pathlib import Path
import sys

try:
    from .run_gtp_subset import load_manifest, uid_pair, check_manifest
except ImportError:
    from run_gtp_subset import load_manifest, uid_pair, check_manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("journal", nargs="?")
    ap.add_argument("--manifest")
    ap.add_argument("--manifest-info")
    ap.add_argument("--arm")
    ap.add_argument("--model")
    ap.add_argument("--suite")
    a = ap.parse_args(argv)
    if a.manifest_info:
        m = load_manifest(a.manifest_info)
        check_manifest(m, a.model, a.suite)
        print(len(m["selected"]), m["sha256"])
        return
    if not a.journal:
        ap.error("journal is required")
    m = load_manifest(a.manifest) if a.manifest else None
    u, rows = {}, 0
    try:
        with open(a.journal) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                rows += 1
                if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
                    uid = r["task_uid"]
                    if a.arm and uid.split(":", 1)[0] != a.arm:
                        continue
                    p = uid_pair(uid)
                    if m and p not in m["selected"]:
                        continue
                    u[p if m else uid] = r
    except FileNotFoundError:
        pass
    print(len(u), sum(1 for r in u.values() if r.get("success")), rows)


if __name__ == "__main__":
    main()
