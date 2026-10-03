"""E3 preliminary: compact per-control physical series from R6 P3 v2 client telemetry (read-only).

For each selected arm/episode, parse client_telemetry/<uid>_aN/controls.jsonl once and write a small npz with
exactly the fields a failure-forensics tool needs (eef, gripper, object poses, goal predicates, entity-level
contacts, issued action, source) plus determinism probes (sim_state digests, before==previous after).
Also records the JSON byte share of every top-level field so the debug-mode size budget can be estimated.

Usage (repo root):
  taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 ... .venv/bin/python -m \
    exp.offline_search.rounds.r08.ideation.E3_failure_forensics.extract_compact --cells pi05_l10_50 --arms A P10
"""
import argparse
import collections
import hashlib
import json
import pathlib

import numpy as np

RUNS = pathlib.Path("/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/runs")
OUT = pathlib.Path("/tmp/r8_E3_failure_forensics/compact")
ROBOT_PREFIX = ("robot0_", "gripper0_", "mount0_")


def digest(x):
    return hashlib.sha1(np.asarray(x, dtype=np.float64).tobytes()).hexdigest()[:16]


def entity_of(geom, objects):
    if geom is None:
        return "other"
    if geom in ("gripper0_finger1_pad_collision",):
        return "pad1"
    if geom in ("gripper0_finger2_pad_collision",):
        return "pad2"
    if geom.startswith(ROBOT_PREFIX):
        return "robot"
    for o in objects:
        if geom.startswith(o + "_"):
            return o
    if "table" in geom or geom in ("floor",):
        return "table"
    return "other:" + geom


def fixture_bodies(bodies):
    out = []
    for b in bodies:
        if b.startswith(ROBOT_PREFIX) or b in ("world", "floor") or "table" in b:
            continue
        out.append(b)
    return out


def parse_episode(path, measure=True):
    objects, preds, geoms, bodies = None, None, None, None
    rows = []
    sizes = collections.Counter()
    decisions = []
    end = None
    prev_after = None
    det = dict(before_eq_prev_after=0, before_ne_prev_after=0)
    init_digest = None
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            ev = j["ev"]
            sizes["line:" + ev] += len(line)
            if ev == "reset":
                ini = j["initial"]
                geoms = ini["entity_ids"]["geom_names"]
                bodies = ini["entity_ids"]["body_names"]
                num = ini["observation_numeric"]
                objects = sorted(k[:-4] for k in num if k.endswith("_pos") and not k.startswith("robot0")
                                 and not k.endswith("_to_robot0_eef_pos"))
                preds = sorted(ini["predicates"].keys()) if isinstance(ini["predicates"], dict) else []
                prev_after = digest(ini["sim_state"])
                init_digest = prev_after
                fixtures = [b for b in fixture_bodies(bodies) if not any(b.startswith(o) for o in objects)]
                continue
            if ev == "decision":
                decisions.append(dict(step=j["decision_step"], control=j["control"],
                                      source=(j.get("marker") or {}).get("source"),
                                      state=j.get("observation_state")))
                continue
            if ev == "rollout_end":
                end = dict(success=j.get("success"), controls=j.get("controls"),
                           final_env_timestep=j.get("final_env_timestep"),
                           reason=(j.get("timing") or {}).get("termination_reason"))
                continue
            if ev != "control":
                continue
            for k, v in (j.items() if measure else ()):
                if k in ("before", "after"):
                    for kk, vv in v.items():
                        sizes[k + "." + kk] += len(json.dumps(vv))
                else:
                    sizes["control." + k] += len(json.dumps(v))
            a, b = j["after"], j["before"]
            bd = digest(b["sim_state"])
            det["before_eq_prev_after" if bd == prev_after else "before_ne_prev_after"] += 1
            ad = digest(a["sim_state"])
            prev_after = ad
            num = a["observation_numeric"]
            # NOTE: P3 v2 entity_ids.geom_names is NOT index-aligned with contact geom ids (155 names, ids up to
            # >=158; geom 10 = table in practice). Contacts are therefore only counted, not named.
            ncon = len(a["contacts"])
            bx = np.asarray(a["body_xpos"], dtype=np.float64)
            pr = a["predicates"]
            rows.append(dict(
                control=j["control"], dec=-1 if j["decision_step"] is None else j["decision_step"],
                source=str(j.get("source")), wait=bool(j.get("wait_phase")),
                act=np.asarray(j["action_issued"], dtype=np.float64)[:7],
                eef=np.asarray(num["robot0_eef_pos"]), eefq=np.asarray(num["robot0_eef_quat"]),
                grip=np.asarray(num["robot0_gripper_qpos"]),
                obj=np.stack([np.asarray(num[o + "_pos"]) for o in objects]),
                objq=np.stack([np.asarray(num[o + "_quat"]) for o in objects]),
                fix=np.stack([bx[bodies.index(fb)] for fb in fixtures]) if fixtures else np.zeros((0, 3)),
                pred=np.array([bool(pr.get(p)) is True for p in preds]),
                ncon=ncon,
                digest=ad, done=bool(j["done"]),
            ))
    return dict(objects=objects, preds=preds, fixtures=fixtures, rows=rows, sizes=sizes, decisions=decisions,
                end=end, det=det, init_digest=init_digest)


def save(ep, out):
    r = ep["rows"]
    stack = lambda k: np.stack([x[k] for x in r])  # noqa: E731
    np.savez_compressed(out, objects=np.array(ep["objects"]), preds=np.array(ep["preds"]),
                        fixtures=np.array(ep["fixtures"]),
                        control=stack("control"), dec=stack("dec"), source=np.array([x["source"] for x in r]),
                        wait=stack("wait"), act=stack("act"), eef=stack("eef"), eefq=stack("eefq"),
                        grip=stack("grip"), obj=stack("obj"), objq=stack("objq"), fix=stack("fix"),
                        pred=stack("pred"), ncon=stack("ncon"), digest=np.array([x["digest"] for x in r]),
                        done=stack("done"), end=json.dumps(ep["end"]), det=json.dumps(ep["det"]),
                        init_digest=ep["init_digest"], decisions=json.dumps(ep["decisions"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+", required=True)
    p.add_argument("--arms", nargs="+", default=["A", "P10"])
    p.add_argument("--reps", nargs="+", default=["r0", "r1", "r2"])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total = collections.Counter()
    for cell in a.cells:
        for arm in a.arms:
            for rep in a.reps:
                name = f"r6p3v2_{cell}_{arm}_{rep}"
                root = RUNS / name / "client_telemetry"
                if not root.exists():
                    print("missing", name, flush=True)
                    continue
                for d in sorted(root.iterdir()):
                    f = d / "controls.jsonl"
                    out = OUT / f"{name}__{d.name}.npz"
                    if out.exists() or not f.exists():
                        continue
                    ep = parse_episode(f, measure=(len(total) == 0))
                    if not ep["rows"]:
                        continue
                    save(ep, out)
                    total.update(ep["sizes"])
                print("done", name, flush=True)
    sz = OUT.parent / f"sizes_{'_'.join(a.cells)}.json"
    sz.write_text(json.dumps(dict(total), indent=1))


if __name__ == "__main__":
    main()
