import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from exp.offline_search.closed_loop.ops.h100 import assets, control, node
from exp.offline_search.closed_loop.ops.remote import count
from exp.offline_search.closed_loop.ops.h100.fleet import FLEETS, island, worker_host


@pytest.mark.parametrize("host,cap,gpus,path", [
    ("timan108", 40, 4, "/scratch/zixuans8/openpi_trace"),
    ("timan107", 64, 8, "/scratch/zixuans8/openpi_trace_h100"),
])
def test_fleet_selection_paths_and_caps(monkeypatch, host, cap, gpus, path):
    monkeypatch.setenv("WORKER_HOST", host)
    assert worker_host() == host and str(island()) == path
    assert FLEETS[host][1] == tuple(range(gpus))
    assert control.REMOTE_NODE[host] == path + "/os_cl/node.py"
    monkeypatch.setenv("PORTS", "23220")
    monkeypatch.setenv("WPS", str(cap))
    assert control.ports_workers() == ([23220], cap)
    monkeypatch.setenv("WPS", str(cap + 1))
    with pytest.raises(ValueError, match=f"1..{cap}"):
        control.ports_workers()


def test_fleet_default_and_invalid_host(monkeypatch):
    monkeypatch.delenv("WORKER_HOST", raising=False)
    assert worker_host() == "timan108"
    monkeypatch.setenv("WORKER_HOST", "h100")
    with pytest.raises(ValueError, match="WORKER_HOST"):
        control.ports_workers()


@pytest.mark.parametrize("host,gpus,cap", [("timan108", "0,1,2,3", 40), ("timan107", "0,1,2,3,4,5,6,7", 64)])
def test_run_arm_uses_host_gpu_cycle_and_cap(tmp_path, host, gpus, cap):
    fake = tmp_path / "python"
    fake.write_text('#!/bin/bash\nif [ "$1" = - ]; then echo "5 default"; else printf "%s\\n" "$@"; fi\n')
    fake.chmod(0o755)
    source = Path(node.__file__).with_name("run_arm.sh").read_text()
    source = source.replace("/scratch/zixuans8/openpi_trace_h100", str(tmp_path)).replace("/scratch/zixuans8/openpi_trace", str(tmp_path))
    source = source.replace("/scratch/zixuans8/openpi/.venv/bin/python", str(fake))
    script = tmp_path / "run_arm.sh"
    script.write_text(source)
    env = {**os.environ, "WORKER_HOST": host}
    result = subprocess.run(["bash", str(script), "libero_spatial", "sample", "host:23220", str(cap), str(tmp_path / "out")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[args.index("--gpu-ids") + 1] == gpus
    assert args[args.index("--gpus") + 1] == str(len(gpus.split(",")))
    assert args[args.index("--workers") + 1] == str(cap)
    result = subprocess.run(["bash", str(script), "libero_spatial", "sample", "host:23220", str(cap + 1), str(tmp_path / "out")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2 and f"worker cap: 1..{cap}" in result.stderr


def test_worker_only_setup_never_contacts_h100_or_other_fleet(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKER_HOST", "timan107")
    monkeypatch.setattr(control, "source_files", lambda: [])
    contacts, bundles = [], []
    monkeypatch.setattr(control, "remote", lambda host, *a, **k: contacts.append(host))
    monkeypatch.setattr(control, "push", lambda host, *a, **k: contacts.append(host))
    def bundle(host, dest, files):
        contacts.append(host)
        bundles.append((dest, {str(rel): Path(src).read_text() for src, rel in files}))
    monkeypatch.setattr(control, "bundle", bundle)
    control._setup(worker_only=True)
    assert set(contacts) == {"timan107"}
    dest, files = bundles[0]
    assert dest == island("timan107")
    assert str(island("timan107")) in files["os_cl/run_gtp_subset.py"]
    assert 'sys.path.insert(0, "/scratch/zixuans8/openpi_trace")' not in files["os_cl/run_gtp_subset.py"]
    for suite in ("libero_spatial", "libero_10"):
        cfg = yaml.safe_load(files[f"exp/ablation_study/cache_size/config/apool_{suite}.yaml"])
        assert cfg["apool_dir"].startswith(str(dest) + "/")


def fixture_run(tmp_path):
    root, store, base = tmp_path / "local/run", tmp_path / "local/store", tmp_path / "remote"
    (root / "config").mkdir(parents=True)
    (root / "fits").mkdir()
    row = dict(arm="sample", mode="plugin", model="pi05", suite="libero_spatial", cell="pi05_spatial_cache",
               method="some.module:Method", kwargs=dict(model_path=str(root / "fits/cal.json")),
               yaml=str(root / "config/sample.yaml"), matrix=str(root / "config/matrix_sample.yaml"),
               plugin_args=["--os-root", str(store), "--os-fit-artifact", str(root / "fits/sample.pkl")])
    (root / "fits/cal.json").write_text('{"value":1}')
    blob = dict(spec=row["method"], kwargs=copy.deepcopy(row["kwargs"]), cell=row["cell"],
                method=SimpleNamespace(cand_name="current"), registered={})
    (root / "fits/sample.pkl").write_bytes(pickle.dumps(blob))
    library = tmp_path / "native.pkl"
    library.write_bytes(b"payload")
    (root / "config/sample.yaml").write_text(yaml.safe_dump(dict(backend=dict(in_memory=dict(preload_path=str(library))))))
    (root / "config/matrix_sample.yaml").write_text("arms: []\n")
    (root / "arms.json").write_text(json.dumps([row]))
    d = store / "library/pi05_spatial/current"
    d.mkdir(parents=True)
    for filename in ("action.npy", "task_id.npy", "ids.json"):
        (d / filename).write_bytes(b"content")
    (d / "manifest.json").write_text(json.dumps(dict(sources=dict(pkl=str(library)))))
    (d / "tok").mkdir()
    (d / "tok/v0.npy").write_bytes(b"must never transfer token corpus")
    big = d.parent / "bpool_all"
    big.mkdir()
    for filename in ("action.npy", "task_id.npy", "key_v0.npy"):
        (big / filename).write_bytes(b"content")
    return root, store, base, row


def test_prepare_second_run_preserves_inputs_and_uses_fresh_outputs(tmp_path):
    from exp.offline_search.closed_loop.ops.h100.prepare_run import prepare
    source, store, base, row = fixture_run(tmp_path)
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    destination = tmp_path / "second"
    prepare(source, destination, ["sample"])
    rows = json.loads((destination / "arms.json").read_text())
    assert rows[0]["yaml"] == str(destination / "config/sample.yaml")
    assert rows[0]["matrix"] == str(destination / "config/matrix_sample.yaml")
    assert rows[0]["plugin_args"] == row["plugin_args"]
    assert (destination / "config/sample.yaml").read_bytes() == Path(row["yaml"]).read_bytes()
    assert json.loads((destination / "eval500.json").read_text()) == [[t, i] for t in range(10) for i in range(50)]
    assert not (destination / "state").exists() and not (destination / "runs").exists()
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="fresh destination"):
        prepare(source, destination, ["sample"])


def test_plan_exact_online_dependencies_and_relocated_metadata(tmp_path, monkeypatch):
    root, store, base, row = fixture_run(tmp_path)
    # Fixture's native PKL lives outside production prefixes: map it explicitly.
    original = assets.remap
    def mapper(v, r, s=store, b=base):
        if isinstance(v, str) and v == str(tmp_path / "native.pkl"):
            return str(base / "native.pkl")
        return original(v, r, s, b)
    monkeypatch.setattr(assets, "remap", mapper)
    initial = (root / "arms.json").read_bytes()
    plan = assets.build_plan(root, ["sample"], root / "plan", store=store, base=base)
    assert not any("tok/" in f["rel"] or "key_v1" in f["rel"] for f in plan["files"])
    assert any(f["rel"].endswith("bpool_all/key_v0.npy") for f in plan["files"])
    assert sum(f["size"] for f in plan["files"]) == plan["bytes"]
    assert (root / "arms.json").read_bytes() == initial
    fit = next(f for f in plan["files"] if f["original"].endswith("sample.pkl"))
    with open(fit["source"], "rb") as f:
        blob = pickle.load(f)
    assert blob["kwargs"] == plan["arms"][0]["kwargs"]
    assert fit["source_sha256"] != fit["sha256"]
    assert any(f["rel"].endswith("bpool_all/action.npy") for f in plan["files"])


def test_rewrite_nested_paths_and_root_boundaries(tmp_path):
    root = Path("/home/weiland/trace_runs/os_closed_loop/run")
    val = {"p": "<RUN>/fits/a.pkl", "store": str(assets.STORE) + "/library/pi05_l10",
           "external": "/home/weiland/trace_runs/os_closed_loop/r05_x/fits/a.pkl", "n": 5}
    out = assets.remap(val, root)
    assert out["p"] == str(assets.BASE / "runs/run/fits/a.pkl")
    assert out["store"] == str(assets.BASE / "store/library/pi05_l10")
    assert out["external"].startswith(str(assets.BASE / "mirror"))
    assert assets.remap("exp.module:Method", root) == "exp.module:Method"
    assert val["p"] == "<RUN>/fits/a.pkl"


def test_clip_online_weights_are_planned_and_sha_checked(tmp_path, monkeypatch):
    root, store, base, row = fixture_run(tmp_path)
    weight = root / "fits/clip.safetensors"
    weight.write_bytes(b"clip image tower")
    module = "exp.offline_search.rounds.r08.abl.clip.encoder"
    monkeypatch.setitem(os.sys.modules, module, SimpleNamespace(weights_path=lambda: weight, WEIGHT_SHA256=node.sha(weight)))
    row["method"] = "exp.offline_search.rounds.r08.abl.clip.method:ClipAWM"
    (root / "arms.json").write_text(json.dumps([row]))
    p = root / "fits/sample.pkl"
    blob = pickle.loads(p.read_bytes())
    blob["spec"] = row["method"]
    p.write_bytes(pickle.dumps(blob))
    original = assets.remap
    def mapper(v, r, s=store, b=base):
        if isinstance(v, str) and v == str(tmp_path / "native.pkl"):
            return str(base / "native.pkl")
        return original(v, r, s, b)
    monkeypatch.setattr(assets, "remap", mapper)
    plan = assets.build_plan(root, ["sample"], root / "plan", store=store, base=base)
    assert any(x["original"] == str(weight) for x in plan["files"])
    assert plan["arms"][0]["server_env"]["R8_CLIP_WEIGHTS"] == str(base / "runs/run/fits/clip.safetensors")


def test_worker_cap_and_port_validation(monkeypatch):
    monkeypatch.setenv("PORTS", "23210,23211,23212,23213")
    monkeypatch.setenv("WPS", "8")
    assert control.ports_workers() == ([23210, 23211, 23212, 23213], 8)
    monkeypatch.setenv("WPS", "10")
    assert control.ports_workers() == ([23210, 23211, 23212, 23213], 10)
    monkeypatch.setenv("WPS", "11")
    with pytest.raises(ValueError, match="1..40"):
        control.ports_workers()
    monkeypatch.setenv("PORTS", "23198")
    with pytest.raises(ValueError, match="23200"):
        control.ports_workers()


def test_selection_manifest_precedes_cartesian_and_task_only(tmp_path, monkeypatch):
    row = dict(model="pi05", suite="libero_spatial")
    monkeypatch.setenv("OSCL_TASKS", "0,1")
    monkeypatch.delenv("OSCL_MANIFEST", raising=False)
    monkeypatch.delenv("OSCL_EPISODES", raising=False)
    assert control.selection(tmp_path, row)[0] == 100
    monkeypatch.setenv("OSCL_EPISODES", "0,1")
    assert control.selection(tmp_path, row)[0] == 4
    p = tmp_path / "manifest.json"
    p.write_text("[[0,0]]")
    monkeypatch.setenv("OSCL_MANIFEST", str(p))
    assert control.selection(tmp_path, row)[0] == 1


def test_completion_dedup_purge_and_resume(tmp_path, capsys):
    journal = tmp_path / "journal.jsonl"
    def record(uid, **extra):
        return dict(task_uid=uid, accepted=True, status="done", run_id="r", attempt=1, **extra)
    good = record("sample:eval:0:0", success=True)
    bad = record("sample:eval:0:1", success=False)
    err = record("sample:eval:1:1", error="error")
    journal.write_text("\n".join(json.dumps(r) for r in (good, good, bad, err)) + "\n")
    (tmp_path / "per_step.jsonl").write_text(json.dumps(dict(_kind="client_timing", termination_reason="exception",
        task_uid=bad["task_uid"], run_id="r", attempt=1)) + "\n")
    count.main([str(journal), "--arm", "sample"])
    assert capsys.readouterr().out.strip() == "2 1 4"
    purge = Path(count.__file__).with_name("purge_exc.py")
    assert subprocess.check_output([os.sys.executable, str(purge), str(tmp_path)], text=True).strip() == "1"
    count.main([str(journal), "--arm", "sample"])
    assert capsys.readouterr().out.strip() == "1 1 3"
    assert (tmp_path / "purged_exceptions.jsonl").exists()


def test_foreign_session_is_never_adopted(tmp_path, monkeypatch):
    monkeypatch.setattr(node, "BASE", tmp_path)
    monkeypatch.setattr(node, "session_exists", lambda receipt: True)
    spec = dict(role="server", out=str(tmp_path / "runs/x"), tag="x", port=23210)
    with pytest.raises(RuntimeError, match="occupied session"):
        node.launch(spec)


def test_duplicate_launch_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(node, "BASE", tmp_path)
    out = tmp_path / "runs/x"
    out.mkdir(parents=True)
    spec = dict(role="server", out=str(out), tag="x", port=23210)
    (out / "server_x.owner.json").write_text(json.dumps(dict(spec=spec)))
    monkeypatch.setattr(node, "session_exists", lambda receipt: True)
    monkeypatch.setattr(node, "ours", lambda receipt: True)
    # Would fail if an idempotent call tried to inspect an occupied port or start another process.
    node.launch(spec)


def test_late_duplicate_after_stop_never_reopens_port(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(node, "BASE", tmp_path)
    out = tmp_path / "runs/x"
    out.mkdir(parents=True)
    spec = dict(role="server", out=str(out), tag="x", port=23210, launch_id="request1")
    (out / "server_x.owner.json").write_text(json.dumps(dict(spec=spec, stopped=True)))
    monkeypatch.setattr(node, "session_exists", lambda receipt: False)
    monkeypatch.setattr(node, "port_busy", lambda _: pytest.fail("a stopped request must never relaunch"))
    node.launch(spec)
    assert capsys.readouterr().out.strip() == "ALREADY_FINISHED oscl23210"


def test_supervisor_records_natural_completion(tmp_path):
    receipt = tmp_path / "owner.json"
    receipt.write_text(json.dumps(dict(spec={})))
    job = dict(cmd=[os.sys.executable, "-c", "print('completed')"], env={}, cwd=str(tmp_path),
               log=str(tmp_path / "server.log"), pidfile=str(tmp_path / "pid.json"), receipt=str(receipt), role="server")
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job))
    # Run in a child so the supervisor's signal handlers do not affect pytest.
    subprocess.run([os.sys.executable, str(Path(node.__file__)), "supervise", str(p)], check=True)
    assert json.loads(receipt.read_text())["finished"] is True
    assert "SERVER_EXIT=0" in (tmp_path / "server.log").read_text()


def test_sha_and_duplicate_json_reply(tmp_path):
    p = tmp_path / "data"
    p.write_bytes(b"data")
    assert node.sha(p) == "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7"
    assert control.json_output('{"running":true}\n{"running":true}') == {"running": True}


def test_full_model_need_and_single_replica(tmp_path, monkeypatch):
    monkeypatch.delenv("NEED_MB", raising=False)
    row = dict(arm="sample", mode="plugin", model="pi05", suite="libero_spatial", cell="pi05_spatial_cache",
               yaml="config.yaml", method="module:Method", full_model=True)
    spec = control.server_spec(tmp_path, row, 23210)
    assert spec["need_mb"] == 9000 and spec["env"]["STAGE1_ONLY"] == "0"
    cmd, env = node.command("pi05", "libero_spatial", 23210, "config.yaml", "/data/oscl_h100/runs/x", "x", [], {})
    assert cmd[cmd.index("--replicas")+1] == "1"


def test_disk_reserve_blocks_before_transfer(tmp_path, monkeypatch):
    monkeypatch.setattr(node.os, "statvfs", lambda p: SimpleNamespace(f_bavail=30, f_frsize=1 << 30))
    monkeypatch.setattr(node, "call", lambda *a, **k: pytest.fail("must not transfer below reserve"))
    with pytest.raises(RuntimeError, match="DISK_BLOCKED"):
        node.pull_assets(dict(dest=str(tmp_path), url="rsync://example", files=[dict(rel="a", size=1)]))


def test_resumable_daemon_pull_checks_exact_content(tmp_path, monkeypatch):
    # Exercise the actual unprivileged daemon on a verified free topology port.
    import socket
    for port in range(23180, 23198):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            break
    else:
        pytest.skip("no free test rsync port in 23180..23197")
    monkeypatch.setenv("SYNC_PORT", str(port))
    monkeypatch.setenv("SYNC_HOST", "127.0.0.1")
    work = tmp_path / "work"
    work.mkdir()
    source = tmp_path / "source"
    source.write_bytes(b"resumable payload")
    item = dict(rel="nested/file", source=str(source), size=source.stat().st_size, sha256=node.sha(source))
    with control.daemon(work, [item]) as url:
        target = tmp_path / "target"
        node.pull_assets(dict(dest=str(target), url=url, files=[item]))
        assert (target / item["rel"]).read_bytes() == source.read_bytes()
        (target / item["rel"]).write_bytes(b"x" * item["size"])
        node.pull_assets(dict(dest=str(target), url=url, files=[item]))
        assert (target / item["rel"]).read_bytes() == source.read_bytes()


@pytest.mark.parametrize("collect_fails", [False, True])
def test_chain_resume_markers_and_collection_failure(tmp_path, monkeypatch, collect_fails):
    root = tmp_path / "run"
    root.mkdir()
    row = dict(arm="sample", model="pi05", suite="libero_spatial", mode="stock", yaml="cfg.yaml")
    (root / "arms.json").write_text(json.dumps([row]))
    (root / "h100_sync").mkdir()
    (root / "h100_sync/synced.json").write_text(json.dumps(dict(arms=[row], arms_sha256=node.sha(root / "arms.json"))))
    monkeypatch.setenv("PORTS", "23210")
    monkeypatch.setenv("WPS", "1")
    monkeypatch.setenv("OSCL_TASKS", "0,1")
    monkeypatch.setenv("OSCL_EPISODES", "0,1")
    monkeypatch.delenv("OSCL_MANIFEST", raising=False)
    monkeypatch.setattr(control.time, "sleep", lambda _: None)
    monkeypatch.setattr(control.signal, "signal", lambda *args: None)
    monkeypatch.setattr(control, "CHAIN_LOCK", tmp_path / "locks/h100_chain.lock")
    starts, stopped = [], []
    def rpc(machine, action, spec, **kw):
        if action == "manifest":
            return json.dumps(dict(sha256="a" * 64, files=1))
        if action == "start":
            starts.append(spec["role"])
            return "STARTED"
        if action == "stop":
            stopped.append(spec["role"])
            return "STOPPED"
        return json.dumps(dict(running=spec["role"] == "server", listening=True, dead=False))
    monkeypatch.setattr(control, "rpc", rpc)
    counts = iter(["3 1 3", "4 2 4"])
    def remote(machine, cmd):
        if machine == "h100":
            return "81000"
        return "0" if "purge_exc.py" in str(cmd) else next(counts)
    monkeypatch.setattr(control, "remote", remote)
    def collect(*args):
        if collect_fails:
            raise RuntimeError("archive SHA mismatch")
        return dict(complete=4)
    monkeypatch.setattr(control, "collect", collect)
    control.chain(root, ["sample"])
    if collect_fails:
        assert (root / "state/sample.ERROR").exists()
        assert (root / "state/CHAIN.ERROR").exists()
        assert not (root / "state/sample.DONE").exists()
        assert not (root / "state/CHAIN.DONE").exists()
        assert "COLLECT_PENDING arm=sample" in (root / "runs/chain.log").read_text()
    else:
        assert (root / "state/sample.DONE").exists()
        assert (root / "state/CHAIN.DONE").exists()
    assert starts.count("driver") == 2
    assert stopped == ["driver", "driver", "server"]
    assert (root / "state/current").read_text() == "none\n"
    assert "ARM_INCOMPLETE" in (root / "runs/chain.log").read_text()
