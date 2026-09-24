"""Resume and merge the interrupted ``init_probe_pilot_20260922`` cohort (2026-09-22 handoff).

The pilot stopped at 12/40 episodes to release devices. The resume keeps the original
worker out root, run plans and journals (the conductor skips accepted terminals) and
serves from a NEW evidence segment. Three subcommands:

* ``server-cmd <policy> <segment>``: the pilot server command from the saved contract argv,
  with only ``--diag-out``, ``--launch-id`` and ``--probe-out`` pointed at the new segment.
* ``check <policy> <segment>``: the new segment's manifest must carry the pilot config_sha
  and the same wrapper source hash / probe contract.
* ``merge <policy> <segment> <final_client_dir>``: an identity-checked merged view
  ``merged/<policy>/{probe,server,client}`` that the unchanged analyzer can read.
  Pilot evidence enters only for the accepted terminals of the stopped snapshot
  (``client_stopped``); the new segment only for episodes accepted after it. Arrays are
  copied under a segment prefix with their bytes (and SHA256) unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path("exp/step_diag/data/init_probe_20260922")
PILOT = ROOT / "pilot"
EXPERIMENT = "init_probe_pilot_20260922"
PORTS = {"groot": 23158, "pi05": 23147}
PY = {"groot": "/home/weiland/gr00t_n15_venv/.venv/bin/python", "pi05": ".venv/bin/python"}
ENV = {
    "groot": ("PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/projects/openpi/src:/home/weiland/projects/openpi:"
              "/home/weiland/projects/openpi/packages/openpi-client/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
              "NO_ALBUMENTATIONS_UPDATE=1"),
    "pi05": ("PYTHONPATH=/home/weiland/projects/openpi/src:/home/weiland/projects/openpi:"
             "/home/weiland/projects/openpi/packages/openpi-client/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
             "OPENPI_SERVER_GPU_MEMORY_LOCK=0"),
}
COMMON_ENV = "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace(argv: list[str], flag: str, value: str) -> list[str]:
    if argv.count(flag) != 1:
        raise ValueError(f"expected exactly one {flag} in the contract argv")
    out = list(argv)
    out[out.index(flag) + 1] = value
    return out


def resume_argv(contract: dict, segment_dir: Path, launch_id: str) -> list[str]:
    """Pilot argv with only the evidence location and server launch-id replaced."""
    argv = _replace(contract["argv"], "--diag-out", str(segment_dir / "server"))
    return _replace(argv, "--launch-id", launch_id)


def server_cmd(policy: str, segment: str) -> str:
    contract = json.loads((PILOT / policy / "probe/probe_contract.json").read_text())
    seg = ROOT / segment / policy
    if (seg / "probe/probe_contract.json").exists():
        raise FileExistsError(f"segment already started: {seg}")
    argv = resume_argv(contract, seg, f"{segment}_{PORTS[policy]}")
    probe = ["-m", "exp.step_diag.serve_init_probe", "--policy", policy, "--probe-out", str(seg / "probe"), "--"]
    return f"{ENV[policy]} {COMMON_ENV} {PY[policy]} -u " + shlex.join(probe + argv)


def check(policy: str, segment: str) -> dict:
    old_m = json.loads((PILOT / policy / "server/manifest_shadow.json").read_text())
    seg = ROOT / segment / policy
    new_m = json.loads((seg / "server/manifest_shadow.json").read_text())
    old_c = json.loads((PILOT / policy / "probe/probe_contract.json").read_text())
    new_c = json.loads((seg / "probe/probe_contract.json").read_text())
    problems = []
    if new_m["config_sha"] != old_m["config_sha"]:
        problems.append("config_sha differs")
    for key in ("schema", "policy", "decisions", "N", "T", "executed", "metric", "random_control", "source_sha256"):
        if old_c.get(key) != new_c.get(key):
            problems.append(f"contract {key} differs")
    expected_argv = resume_argv(old_c, seg, f"{segment}_{PORTS[policy]}")
    if new_c["argv"] != expected_argv:
        problems.append("argv differs beyond the evidence location / launch-id")
    return {"policy": policy, "config_sha": new_m["config_sha"], "problems": problems}


def _accepted(journal_dir: Path) -> dict:
    out = {}
    for path in sorted(journal_dir.glob("journal_*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("accepted") and row.get("status") in ("done", "failed"):
                key = (row["task_uid"], int(row.get("attempt", 1)))
                if key in out:
                    raise ValueError(f"duplicate accepted identity: {key}")
                out[key] = row
    return out


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def merge(policy: str, segment: str, final_client: Path, out_root: Path = ROOT / "merged",
          pilot: Path = PILOT, seg_root: Path | None = None) -> dict:
    """Identity-checked merged view of the pilot and one resume segment."""
    seg = (seg_root or ROOT / segment) / policy
    old_accept = _accepted(pilot / policy / "client_stopped")
    final_accept = _accepted(final_client)
    for key, row in old_accept.items():
        if final_accept.get(key) != row:
            raise ValueError(f"stopped-snapshot terminal changed or vanished in the final journal: {key}")
    new_accept = {k: v for k, v in final_accept.items() if k not in old_accept}
    old_uids = {uid for uid, _ in old_accept}
    if any(uid in old_uids for uid, _ in new_accept):
        raise ValueError("an episode completed in the pilot was accepted again in the resume")
    old_m = json.loads((pilot / policy / "server/manifest_shadow.json").read_text())
    new_m = json.loads((seg / "server/manifest_shadow.json").read_text())
    if old_m["config_sha"] != new_m["config_sha"]:
        raise ValueError("segment config_sha differs from the pilot")
    dst = out_root / policy
    if dst.exists():
        raise FileExistsError(f"merged view exists: {dst}")
    for sub in ("probe", "server", "client"):
        (dst / sub).mkdir(parents=True)
    included = {"pilot": sorted(map(list, old_accept)), segment: sorted(map(list, new_accept))}
    excluded = {"pilot": [], segment: []}
    probe_out = []
    for name, src, keep in (("pilot", pilot / policy, old_accept), (segment, seg, new_accept)):
        (dst / "probe" / name).mkdir()
        for row in _rows(src / "probe/probe.jsonl"):
            identity = (row["task_uid"], int(row["attempt"]))
            if identity not in keep:
                excluded[name].append([row["task_uid"], row["attempt"], row["decision_idx"], row["arrays"]])
                continue
            source = src / "probe" / row["arrays"]
            if sha256(source) != row["arrays_sha256"]:
                raise ValueError(f"array checksum mismatch: {source}")
            target = dst / "probe" / name / row["arrays"]
            shutil.copyfile(source, target)
            probe_out.append({**row, "arrays": f"{name}/{row['arrays']}", "segment": name,
                              "segment_arrays": row["arrays"]})
        server_rows = [r for f in sorted((src / "server").glob("rows_*.jsonl")) for r in _rows(f)
                       if (r["task_uid"], int(r["attempt"])) in keep]
        with (dst / "server" / f"rows_{name}.jsonl").open("w") as f:
            for r in server_rows:
                f.write(json.dumps(r) + "\n")
    with (dst / "probe/probe.jsonl").open("w") as f:
        for row in probe_out:
            f.write(json.dumps(row) + "\n")
    shutil.copyfile(pilot / policy / "probe/probe_contract.json", dst / "probe/probe_contract.json")
    shutil.copyfile(pilot / policy / "server/manifest_shadow.json", dst / "server/manifest_shadow.json")
    for path in sorted(final_client.iterdir()):
        if path.name.startswith(("journal_", "launch_", "run_plan_", "summary_")):
            shutil.copyfile(path, dst / "client" / path.name)
    provenance = {
        "experiment_id": EXPERIMENT, "policy": policy, "segments": ["pilot", segment],
        "config_sha": old_m["config_sha"],
        "contracts": {"pilot": json.loads((pilot / policy / "probe/probe_contract.json").read_text()),
                      segment: json.loads((seg / "probe/probe_contract.json").read_text())},
        "server_manifests": {"pilot": old_m, segment: new_m},
        "client_source": str(final_client),
        "client_files_sha256": {p.name: sha256(p) for p in sorted((dst / "client").iterdir())},
        "included_episodes": included, "excluded_probe_rows": excluded,
        "rule": ("pilot evidence only for accepted terminals of client_stopped; the resume segment only for "
                 "episodes first accepted after it; arrays copied byte-identically under a segment prefix"),
    }
    (dst / "merge_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {"policy": policy, "pilot_episodes": len(old_accept), "resume_episodes": len(new_accept),
            "probe_rows": len(probe_out), "excluded_rows": {k: len(v) for k, v in excluded.items()}}


def main() -> None:
    cmd, policy, segment, *rest = sys.argv[1:]
    if cmd == "server-cmd":
        print(server_cmd(policy, segment))
    elif cmd == "check":
        print(json.dumps(check(policy, segment)))
    elif cmd == "merge":
        print(json.dumps(merge(policy, segment, Path(rest[0]))))
    else:
        raise SystemExit(f"unknown subcommand {cmd}")


if __name__ == "__main__":
    main()
