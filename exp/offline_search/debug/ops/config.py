"""Frozen capture config and safely quoted coordinator launch arguments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

from exp.offline_search.debug.schema import SCHEMA_VERSION
from exp.offline_search.debug.transport.protocol import identifier
from exp.offline_search.debug.transport.receiver import atomic_json, digest

REPO_ROOT = Path(__file__).resolve().parents[4]


def capture_code_files():
    """Only code on the collection path freezes an arm's resumable manifest."""
    package = REPO_ROOT / "exp/offline_search/debug"
    paths = [package / "schema.py"]
    for sub in ("server", "client", "transport", "ops"):
        paths.extend(p for p in (package / sub).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".sh") and "tests" not in p.parts)
    serving = REPO_ROOT / "exp/offline_search/closed_loop"
    paths.extend(serving / name for name in ("plugin.py", "blind.py", "stage_overrides.py"))
    paths.extend(serving.glob("serve_*.py"))
    return {str(p.relative_to(REPO_ROOT)): digest(p) for p in sorted(set(paths)) if p.is_file()}


def arm_spec(run, arm):
    identifier(Path(run).name)
    identifier(arm, arm=True)
    rows = json.loads((Path(run) / "arms.json").read_text())
    return next(r for r in rows if r["arm"] == arm)


def capture_config(run, arm):
    spec = arm_spec(run, arm)
    config = dict(campaign=Path(run).name, arm=arm, rawkeys_mod=16, snapshot_mod=16, draws_mod=32,
                  writer_queue_bytes=512 * 1024 * 1024)
    supplied = os.environ.get("OSDEBUG_CONFIG", "{}")
    config.update(json.loads(supplied))
    config.update(spec.get("debug_config", {}))
    config["campaign"], config["arm"] = Path(run).name, arm
    return config


def oracle_enabled(spec):
    return (bool(spec.get("oracle", False)) or "--os-oracle" in spec.get("plugin_args", [])
            or bool(spec.get("r8", {}).get("oracle_client", False))
            or str(spec.get("client_env", {}).get("OSDEBUG_ORACLE", "0")) == "1")


def server_args(run, arm, port):
    spec = arm_spec(run, arm)
    args = ["--os-debug-dir", str(Path(run) / "runs" / arm / "debug" / ("server_" + str(port))),
            "--os-debug-config", json.dumps(capture_config(run, arm), sort_keys=True)]
    if oracle_enabled(spec) and "--os-oracle" not in spec.get("plugin_args", []):
        args.append("--os-oracle")
    return args


def client_env(run, arm):
    spec = arm_spec(run, arm)
    mode = os.environ.get("OSDEBUG_MODE", "stream")
    if mode not in ("stream", "file"):
        raise ValueError("invalid OSDEBUG_MODE")
    env = dict(OSDEBUG_MODE=mode, OSDEBUG_ARM=arm, OSDEBUG_CAMPAIGN=Path(run).name,
               OSDEBUG_CLIENT_DIR="/scratch/zixuans8/openpi_trace/os_cl/debug_client/{}/{}".format(Path(run).name, arm),
               OSDEBUG_STREAM_RUN=Path(run).name, OSDEBUG_STREAM_ARM=arm,
               OSDEBUG_ORACLE="1" if oracle_enabled(spec) else "0", OSDEBUG_STREAM_QUEUE_BYTES=str(8 * 1024 * 1024))
    if mode == "stream":
        ready = json.loads((Path(run) / "state/osdebug_receiver.json").read_text())
        env.update(OSDEBUG_STREAM="{}:{}".format(os.environ.get("OSDEBUG_RECEIVER_HOST", "ziyanglin.com"), ready["port"]),
                   OSDEBUG_STREAM_TOKEN=(Path(run) / "state/osdebug_stream.token").read_text().strip())
    env.update({k: str(v) for k, v in spec.get("client_env", {}).items() if k.startswith("OSDEBUG_")})
    if any("\0" in k or "\0" in v or not k.replace("_", "").isalnum() for k, v in env.items()):
        raise ValueError("invalid client environment")
    return env


def manifest(run, arm):
    code = capture_code_files()
    spec = arm_spec(run, arm)
    result = dict(schema=SCHEMA_VERSION, arm=arm, campaign=Path(run).name, arm_spec=spec,
                  code_sha256=hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(), code_files=code,
                  arms_sha256=digest(Path(run) / "arms.json"), capture_config=capture_config(run, arm),
                  env_seed=7, privileged_oracle=oracle_enabled(spec), sampling="sha256(campaign|task_uid|decision_seq)",
                  deployment_cost="live only; see server meta for model price table",
                  model_provenance="server meta*.json contains checkpoint/library/fit/stage-table/git hashes")
    path = Path(run) / "runs" / arm / "debug/MANIFEST.json"
    if path.exists():
        frozen = json.loads(path.read_text())
        if frozen == result:
            return result
        # Coordinator-side tools (collect/verify/cleanup, ops scripts) may be fixed mid-campaign without changing
        # what is captured: accept drift there, record it, and keep the frozen manifest. Anything else refuses.
        strict = lambda files: {k: v for k, v in files.items() if _data_defining(k)}
        rest = lambda m: {k: v for k, v in m.items() if k not in ("code_sha256", "code_files")}
        if rest(frozen) == rest(result) and strict(frozen.get("code_files", {})) == strict(code):
            changed = sorted(k for k in set(code) | set(frozen.get("code_files", {}))
                             if code.get(k) != frozen.get("code_files", {}).get(k))
            with open(path.with_name("MANIFEST_drift.jsonl"), "a") as f:
                f.write(json.dumps(dict(accepted="coordinator-tool drift only", changed=changed)) + "\n")
            return frozen
        raise ValueError("capture manifest differs from frozen arm; use a new campaign/arm")
    atomic_json(path, result)
    return result


def _data_defining(rel):
    """Files whose change would alter what is captured or served (never tolerated mid-arm)."""
    if rel.startswith(("exp/offline_search/debug/server/", "exp/offline_search/debug/client/",
                       "exp/offline_search/closed_loop/")) or rel == "exp/offline_search/debug/schema.py":
        return True
    return rel in {"exp/offline_search/debug/transport/" + n for n in
                   ("protocol.py", "sink.py", "receiver.py", "dispatch_fence.py", "__init__.py")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--port", type=int)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--server-args", action="store_true")
    group.add_argument("--client-env", action="store_true")
    group.add_argument("--write-client-env", type=Path)
    group.add_argument("--manifest", action="store_true")
    a = ap.parse_args()
    if a.server_args:
        if a.port is None:
            ap.error("--port required")
        sys.stdout.write("".join(s + "\0" for s in server_args(a.run_root, a.arm, a.port)))
    elif a.client_env:
        print(" ".join(k + "=" + shlex.quote(v) for k, v in client_env(a.run_root, a.arm).items()))
    elif a.write_client_env:
        atomic_json(a.write_client_env, client_env(a.run_root, a.arm))
        a.write_client_env.chmod(0o600)
        print(json.dumps(dict(client_env_written=str(a.write_client_env))))
    else:
        manifest(a.run_root, a.arm)
        print(json.dumps(dict(arm=a.arm, capture_manifest="frozen")))


if __name__ == "__main__":
    main()
