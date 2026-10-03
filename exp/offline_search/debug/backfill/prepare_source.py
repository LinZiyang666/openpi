"""Export exact reset/issued controls from old P3 or osdebug client captures.

Use an explicit library-episode mapping; a task/init match alone does not prove
an old A-pool evaluation is a B-pool demonstration. Replay checks remain final.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from exp.offline_search.debug.client.capture import npz_bytes
from exp.offline_search.debug.transport.receiver import atomic_json, digest
from exp.offline_search.debug.schema import read_jsonl


def export(directory, out, library_episode, provenance, rs_space="model"):
    directory, out = Path(directory), Path(out)
    arrays = {}
    skipped = []
    ep = directory / "episode.json"
    if ep.exists():
        meta = json.loads(ep.read_text())
        with np.load(str(directory / "snap_reset.npz"), allow_pickle=False) as reset:
            if "sim_state" not in reset:
                raise ValueError("capture lacks the environment's flat reset state")
            arrays["initial_state"] = reset["sim_state"].copy()
        blocks = []
        for path in sorted(directory.glob("controls_*.npz")):
            with np.load(str(path), allow_pickle=False) as block:
                blocks.append({k: block[k].copy() for k in ("control_idx", "action", "decision_seq")})
        if not blocks:
            raise ValueError("capture lacks executed controls")
        indices = np.concatenate([b["control_idx"] for b in blocks])
        if not np.array_equal(indices, np.arange(meta["n_controls"])):
            raise ValueError("capture control gap")
        arrays["action"] = np.concatenate([b["action"] for b in blocks])
        arrays["decision_seq"] = np.concatenate([b["decision_seq"] for b in blocks]).astype(np.int32)
        seed, suite = meta["env_seed"], meta["suite"]
    else:
        # P3 reset stores the initial flat simulator state; controls have the
        # exact action_issued. Do not use its defective contact name lists.
        rows = list(read_jsonl(directory / "controls.jsonl", skipped))
        start = next(r for r in rows if r.get("ev") == "attempt_start")
        reset = next(r for r in rows if r.get("ev") == "reset")
        controls = [r for r in rows if r.get("ev") == "control"]
        if [r["control"] for r in controls] != list(range(len(controls))):
            raise ValueError("P3 control gap")
        if not any(r.get("ev") == "rollout_end" for r in rows):
            raise ValueError("P3 episode is an incomplete prefix")
        arrays["initial_state"] = np.asarray(reset["initial"]["sim_state"], dtype=np.float64)
        arrays["action"] = np.asarray([r["action_issued"] for r in controls], dtype=np.float64)
        seq = [r.get("decision_step") for r in controls]
        unique = sorted({s for s in seq if s is not None})
        # P3 step numbers need not start at zero; preserve their original list
        # and map to the exported sequential decision axis explicitly.
        mapping = {s: i for i, s in enumerate(unique)}
        arrays["decision_seq"] = np.asarray([-1 if s is None else mapping[s] for s in seq], dtype=np.int32)
        arrays["original_decision_seq"] = np.asarray(unique, dtype=np.int32)
        seed = start["environment_seed"]
        suite = start.get("suite", start.get("experiment", start.get("source_identity", {}).get("experiment")))
        if not suite:
            raise ValueError("source suite unavailable")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError("source output already exists")
    with out.open("xb") as handle:
        handle.write(npz_bytes(arrays))
    result = {str(library_episode): dict(npz=str(out.resolve()), sha256=digest(out), env_seed=int(seed), suite=suite,
                                       provenance=provenance, rs_space=rs_space, source_client=str(directory.resolve()),
                                       skipped_jsonl_lines=skipped)}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--client-episode", type=Path, required=True)
    ap.add_argument("--library-episode", type=int, required=True)
    ap.add_argument("--provenance", required=True, help="why this exact capture is the library demonstration")
    ap.add_argument("--rs-space", choices=("model", "wire"), default="model")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = export(a.client_episode, a.out, a.library_episode, a.provenance, a.rs_space)
    atomic_json(a.out.with_suffix(".source_map.json"), result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
