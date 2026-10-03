"""Coordinator-only real-server parity CLI; capture/replay wire tapes losslessly.

Record reads stored observations only. Replay connects to an already running
local server; this tool never launches a server or a simulator.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import schema
from .writer import atomic_json


def record_tape(store_root, cell, output, episodes=20, max_decisions=0):
    from exp.offline_search.harness import store
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    qc = store.QueryCell(store_root, cell)
    tidx = np.asarray(qc.tok_index)
    selected = [e for e in qc.episodes if np.all(tidx[e["start"]:e["end"]] >= 0)][:episodes]
    if not selected:
        raise ValueError("no complete image-bearing episodes in " + cell)
    camera0, camera1 = qc.tok("img0"), qc.tok("img1")
    entries = []
    for i, episode in enumerate(selected):
        uid = "tape:eval:{}:{}".format(episode["task_id"], episode["init"])
        end = min(episode["end"], episode["start"] + max_decisions) if max_decisions else episode["end"]
        rows = np.arange(episode["start"], end)
        indices = tidx[rows]
        path = "episode_{:04d}.npz".format(i)
        schema.write_npz_block(output / path, dict(img_third=np.asarray(camera0[indices]),
            img_wrist=np.asarray(camera1[indices]), state_wire=np.asarray(qc.raw_state[rows], np.float64),
            prompt=np.asarray([episode["task"]], dtype="<U{}".format(max(512, len(episode["task"]))))))
        entries.append(dict(task_uid=uid, task_id=int(episode["task_id"]), init=int(episode["init"]),
                            attempt=1, dispatch_gen=0, file=path, task=episode["task"], n_decisions=len(rows)))
    atomic_json(output / "tape.json", dict(schema=schema.SCHEMA_VERSION, cell=cell, campaign="parity", episodes=entries))
    return dict(episodes=len(entries), decisions=sum(e["n_decisions"] for e in entries))


def replay_tape(tape, output, host="127.0.0.1", port=23310, bundle="default", expected_echo=False):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    tape, output = Path(tape), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((tape / "tape.json").read_text())
    client = WebsocketClientPolicy(host=host, port=port)
    client.select_bundle(bundle)
    digest = hashlib.sha256()
    count = 0
    for i, episode in enumerate(metadata["episodes"]):
        arrays = schema.read_npz_block(tape / episode["file"])
        epkey = schema.episode_key(episode["task_uid"], episode["attempt"])
        client.episode_start(experiment="osdebug-parity", task=episode["task"], episode_id=episode["init"],
            extra_metadata=dict(task_uid=episode["task_uid"], task_id=episode["task_id"],
                                orig_init_state_idx=episode["init"], attempt=episode["attempt"]))
        actions, ids, echo_status = [], [], []
        try:
            for seq in range(episode["n_decisions"]):
                did = schema.decision_id(epkey, episode["dispatch_gen"], seq)
                obs = {"observation/image": arrays["img_third"][seq], "observation/wrist_image": arrays["img_wrist"][seq],
                    "observation/state": arrays["state_wire"][seq], "prompt": str(arrays["prompt"][0]),
                    "__extra__": dict(decision_id=seq, executed_steps=5),
                    "__debug__": dict(v=1, decision_id=did, episode_key=epkey, task_uid=episode["task_uid"],
                                      attempt=episode["attempt"], dispatch_gen=episode["dispatch_gen"],
                                      decision_seq=seq, t_client_send=0.)}
                response = client.infer(obs)
                echo = response.pop("__debug__", {})
                if expected_echo and (echo.get("decision_id") != did or echo.get("status") != "available"):
                    raise ValueError("invalid debug echo at " + did + ": " + repr(echo))
                action = np.array(response["actions"], copy=True)
                digest.update(str((action.shape, action.dtype)).encode())
                digest.update(action.tobytes())
                actions.append(action); ids.append(did); echo_status.append(echo.get("status", "unverified"))
                count += 1
        finally:
            client.episode_end(success=False)
        schema.write_npz_block(output / "responses_{:04d}.npz".format(i), dict(
            decision_id=np.asarray(ids, dtype="<U80"), actions=np.stack(actions), echo_status=np.asarray(echo_status)))
    result = dict(episodes=len(metadata["episodes"]), decisions=count, actions_sha256=digest.hexdigest())
    atomic_json(output / "report.json", result)
    return result


def compare_outputs(left, right):
    left, right = Path(left), Path(right)
    files_a = sorted(p.name for p in left.glob("responses_*.npz"))
    files_b = sorted(p.name for p in right.glob("responses_*.npz"))
    if not files_a or files_a != files_b:
        raise ValueError("response episode sets differ or are empty")
    decisions = 0
    for filename in files_a:
        a, b = schema.read_npz_block(left / filename), schema.read_npz_block(right / filename)
        if not np.array_equal(a["decision_id"], b["decision_id"]):
            raise ValueError("decision identities differ: " + filename)
        x, y = a["actions"], b["actions"]
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            raise ValueError("action bytes differ: " + filename)
        decisions += len(a["decision_id"])
    return dict(PASS=True, episodes=len(files_a), decisions=decisions, action_bytes_identical=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record")
    record.add_argument("--store-root", default="/home/weiland/trace_runs/offline_search_store")
    record.add_argument("--cell", required=True)
    record.add_argument("--out", type=Path, required=True)
    record.add_argument("--episodes", type=int, default=20)
    record.add_argument("--max-decisions", type=int, default=0)
    replay = sub.add_parser("replay")
    replay.add_argument("--tape", required=True)
    replay.add_argument("--out", required=True)
    replay.add_argument("--host", default="127.0.0.1")
    replay.add_argument("--port", type=int, default=23310)
    replay.add_argument("--bundle", default="default")
    replay.add_argument("--expect-echo", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--off", required=True)
    compare.add_argument("--on", required=True)
    args = parser.parse_args(argv)
    if args.command == "record":
        result = record_tape(args.store_root, args.cell, args.out, args.episodes, args.max_decisions)
    elif args.command == "replay":
        result = replay_tape(args.tape, args.out, args.host, args.port, args.bundle, args.expect_echo)
    else:
        result = compare_outputs(args.off, args.on)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
