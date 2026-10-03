"""Local fault injection only: no tether, GPU, LIBERO, or remote launches."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from exp.offline_search.closed_loop.ops.h100 import control, node


def test_two_fleets_run_concurrently_and_maintenance_stays_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "CHAIN_LOCK", tmp_path / "h100_chain.lock")
    monkeypatch.setenv("WORKER_HOST", "timan108")
    with control.fleet_lock(tmp_path / "A"):
        monkeypatch.setenv("WORKER_HOST", "timan107")
        with control.fleet_lock(tmp_path / "B"):
            with pytest.raises(RuntimeError, match="H100_CHAIN_BUSY"):
                with control.chain_lock():
                    pytest.fail("full maintenance acquired active chain gate")
            with pytest.raises(RuntimeError, match="H100_CHAIN_BUSY"):
                with control.fleet_lock(tmp_path / "C"):
                    pytest.fail("same host must serialize")
        # Idle B's isolated worker setup remains possible while A holds its gate.
        monkeypatch.setattr(control, "_setup", lambda **kw: kw)
        control.setup(worker_only=True)
        monkeypatch.setenv("WORKER_HOST", "timan108")
        with pytest.raises(RuntimeError, match="H100_CHAIN_BUSY"):
            control.setup(worker_only=True)


def test_chain_B_contacts_only_t107_and_h100(tmp_path, monkeypatch):
    root, _, _ = mock_chain(tmp_path, monkeypatch)
    planpath = root / "h100_sync/synced.json"
    plan = json.loads(planpath.read_text())
    plan["worker_host"] = "timan107"
    planpath.write_text(json.dumps(plan))
    monkeypatch.setenv("WORKER_HOST", "timan107")
    monkeypatch.setenv("PORTS", "23220")
    calls = []
    rpc, remote = control.rpc, control.remote
    def record_rpc(host, action, spec, **kw):
        calls.append((host, action, spec))
        return rpc(host, action, spec, **kw)
    def record_remote(host, cmd, **kw):
        calls.append((host, "remote", cmd))
        return remote(host, cmd, **kw)
    monkeypatch.setattr(control, "rpc", record_rpc)
    monkeypatch.setattr(control, "remote", record_remote)
    control.chain(root, ["first"])
    assert {c[0] for c in calls} == {"h100", "timan107"}
    spec = json.loads((root / "runs/first/h100_launch.json").read_text())
    assert spec["worker_host"] == "timan107"
    assert spec["driver"]["out"].startswith("/scratch/zixuans8/openpi_trace_h100/")
    assert spec["driver"]["env"]["WORKER_HOST"] == "timan107"


def test_wrong_synced_fleet_refused_before_remote_calls(tmp_path, monkeypatch):
    root, events, _ = mock_chain(tmp_path, monkeypatch)
    monkeypatch.setenv("WORKER_HOST", "timan107")
    with pytest.raises(RuntimeError, match="synced worker host differs"):
        control.chain(root, ["first"])
    assert events == []


def test_live_legacy_A_allows_B_but_refuses_same_host_and_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "CHAIN_LOCK", tmp_path / "h100_chain.lock")
    monkeypatch.setattr(control, "legacy_chains", lambda **kw: [dict(root=tmp_path / "A", worker_host="timan108")])
    monkeypatch.setenv("WORKER_HOST", "timan107")
    with control.fleet_lock(tmp_path / "B"):
        pass
    with pytest.raises(RuntimeError, match="H100_CHAIN_BUSY"):
        with control.fleet_lock(tmp_path / "elsewhere/A"):
            pass
    monkeypatch.setenv("WORKER_HOST", "timan108")
    with pytest.raises(RuntimeError, match="H100_CHAIN_BUSY"):
        with control.fleet_lock(tmp_path / "B"):
            pass


def test_legacy_lock_holder_authentication_and_fail_closed(tmp_path, monkeypatch):
    lock = tmp_path / "h100_chain.lock"
    lock.touch()
    monkeypatch.setattr(control, "CHAIN_LOCK", lock)
    pid = 123456789
    st = lock.stat()
    table = f"1: FLOCK ADVISORY WRITE {pid} {os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino} 0 EOF\n"
    root = tmp_path / "A"
    (root / "state").mkdir(parents=True)
    (root / "state/h100_chain.lock").touch()
    argv = b"python\0-m\0exp.offline_search.closed_loop.ops.h100.control\0chain\0" + str(root).encode() + b"\0arm\0"
    receipt = root / "state/h100_chain.owner.json"
    receipt.write_text(json.dumps(dict(pid=pid, starttime="time", cmdline=argv.hex())))
    read_text, read_bytes = Path.read_text, Path.read_bytes
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k: table if str(p) == "/proc/locks" else read_text(p, *a, **k))
    def proc_bytes(p, *a, **k):
        if str(p) == f"/proc/{pid}/cmdline":
            return argv
        if str(p) == f"/proc/{pid}/environ":
            return b"WORKER_HOST=timan108\0"
        return read_bytes(p, *a, **k)
    monkeypatch.setattr(Path, "read_bytes", proc_bytes)
    monkeypatch.setattr(control, "identity", lambda p: "time")
    assert control.legacy_chains() == [dict(root=root, worker_host="timan108")]
    monkeypatch.setattr(control, "identity", lambda p: "recycled")
    with pytest.raises(RuntimeError, match="unauthenticated lock holder"):
        control.legacy_chains()


@pytest.mark.parametrize("host", ["timan108", "timan107"])
def test_rpc_host_island_and_allowed_staging(tmp_path, monkeypatch, host):
    monkeypatch.setattr(control, "_RPC_SPECS", set())
    monkeypatch.setattr(control, "_STAGING_READY", set())
    calls, pushed = [], []
    monkeypatch.setattr(control, "remote", lambda node, cmd, *a, **kw: calls.append((node, cmd)) or "SPEC_MISSING")
    monkeypatch.setattr(control, "run", lambda cmd, **kw: pushed.append(list(map(str, cmd))) or "")
    control.rpc(host, "status", dict(role="driver", out="somewhere"))
    assert all(c[0] == host for c in calls)
    assert calls[-1][1][1] == control.REMOTE_NODE[host]
    assert str(control.island(host) / "os_cl/.rpc") in calls[-1][1][-1]
    expected = "/tmp/oscl_sb3_stage/" if host == "timan107" else "/srv/local/zixuans8/oscl_sb_stage/"
    assert pushed[0][-1].startswith(host + ":" + expected)


@pytest.mark.parametrize("conflict", ["sha", "missing_active", "symlink", "active_sha"])
def test_concurrent_pull_refuses_without_writing_existing_files(tmp_path, monkeypatch, conflict):
    monkeypatch.setattr(node, "BASE", tmp_path)
    p = tmp_path / "shared"
    p.write_bytes(b"A")
    item = dict(rel="shared", sha256=node.sha(p), size=1)
    protected = {"shared": item["sha256"]}
    if conflict == "sha":
        item["sha256"] = "b" * 64
    elif conflict == "missing_active":
        p.unlink()
    elif conflict == "symlink":
        target = tmp_path / "target"
        p.replace(target)
        p.symlink_to(target)
    else:
        protected["shared"] = "b" * 64
    before = p.read_bytes() if p.exists() else None
    monkeypatch.setattr(node, "pull_assets", lambda spec: pytest.fail("must refuse before transfer"))
    with pytest.raises(RuntimeError, match="ASSET_"):
        node.pull_new_assets(dict(dest=str(tmp_path), files=[item], protected=protected))
    assert (p.read_bytes() if p.exists() else None) == before


def test_concurrent_pull_reuses_exact_existing_files_and_replays(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(node, "BASE", tmp_path)
    p = tmp_path / "shared"
    p.write_bytes(b"A")
    old = p.stat()
    existing = dict(rel="shared", sha256=node.sha(p), size=1)
    new = dict(rel="new", sha256=hashlib.sha256(b"B").hexdigest(), size=1)
    pulls = []
    def pull(spec):
        pulls.append(spec)
        assert spec["files"] == [new]
        (tmp_path / "new").write_bytes(b"B")
    monkeypatch.setattr(node, "pull_assets", pull)
    spec = dict(dest=str(tmp_path), files=[existing, new], protected={"shared": existing["sha256"]})
    node.pull_new_assets(spec)
    node.pull_new_assets(spec)
    assert len(pulls) == 1 and (p.stat().st_ino, p.stat().st_mtime_ns) == (old.st_ino, old.st_mtime_ns)
    assert capsys.readouterr().out == "ASSETS_NEW_OK added=1 reused=1\nASSETS_NEW_OK added=0 reused=2\n"


def test_concurrent_sync_conflicting_active_plan_refuses_before_remote_mutation(tmp_path, monkeypatch):
    root = tmp_path / "B"
    root.mkdir()
    (root / "arms.json").write_text("[]")
    active = tmp_path / "A/h100_sync"
    active.mkdir(parents=True)
    (active / "synced.json").write_text(json.dumps(dict(files=[dict(rel="shared", sha256="a" * 64)])))
    monkeypatch.setattr(control, "build_plan", lambda *a: dict(files=[dict(rel="shared", sha256="b" * 64, size=1)]))
    monkeypatch.setattr(control, "rpc", lambda *a, **k: pytest.fail("no remote mutation allowed"))
    with pytest.raises(RuntimeError, match="ACTIVE_ASSET_CONFLICT"):
        control._sync(root, ["arm"], concurrent=True, active=[dict(root=active.parent)])


def test_concurrent_sync_B_while_A_uses_shared_assets(tmp_path, monkeypatch):
    from contextlib import contextmanager
    a, b = tmp_path / "A", tmp_path / "B"
    (a / "h100_sync").mkdir(parents=True)
    b.mkdir()
    cfg, matrix = b / "cfg.yaml", b / "matrix.yaml"
    cfg.write_text("cfg")
    matrix.write_text("matrix")
    row = dict(arm="sample", yaml=str(cfg), matrix=str(matrix))
    (b / "arms.json").write_text(json.dumps([row]))
    item = dict(rel="shared", sha256=node.sha(cfg), size=3, source=str(cfg), original=str(cfg))
    (a / "h100_sync/synced.json").write_text(json.dumps(dict(files=[item])))
    monkeypatch.setattr(control, "CHAIN_LOCK", tmp_path / "h100_chain.lock")
    monkeypatch.setattr(control, "build_plan", lambda *args: dict(files=[item], arms=[row]))
    @contextmanager
    def daemon(*args):
        yield "rsync://mock"
    monkeypatch.setattr(control, "daemon", daemon)
    calls = []
    monkeypatch.setattr(control, "rpc", lambda host, action, spec, **kw: calls.append((host, action, spec)) or "OK")
    monkeypatch.setattr(control, "bundle", lambda host, dest, pairs: calls.append((host, "bundle", dest)))
    monkeypatch.setenv("WORKER_HOST", "timan108")
    with control.fleet_lock(a):
        monkeypatch.setattr(control, "legacy_chains", lambda **kw: [dict(root=a, worker_host="timan108")])
        monkeypatch.setenv("WORKER_HOST", "timan107")
        control.sync(b, ["sample"], concurrent=True)
        with pytest.raises(RuntimeError, match="H100_CHAIN_BUSY"):
            control.sync(b, ["sample"])
    assert calls[0][0:2] == ("h100", "pull-new")
    assert calls[0][2]["protected"] == {"shared": node.sha(cfg)}
    assert calls[1] == ("timan107", "bundle", control.island("timan107") / "os_cl")
    assert json.loads((b / "h100_sync/synced.json").read_text())["worker_host"] == "timan107"


def test_old_launch_receipts_route_abort_collection_to_t108(tmp_path, monkeypatch):
    path = tmp_path / "runs/arm/h100_launch.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(driver={})))
    monkeypatch.setenv("WORKER_HOST", "timan107")
    assert control.launch_host(tmp_path, "arm") == "timan108"
    path.write_text(json.dumps(dict(worker_host="timan107")))
    assert control.launch_host(tmp_path, "arm") == "timan107"


def test_concurrent_sync_uses_only_versioned_h100_helper(monkeypatch):
    monkeypatch.setattr(control, "_RPC_SPECS", set())
    calls = []
    monkeypatch.setattr(control, "remote", lambda host, cmd, *a, **k: calls.append(cmd) or "SPEC_EXISTS")
    control.rpc("h100", "pull-new", dict(dest=str(control.BASE), files=[]))
    assert calls[-1][1] == str(control.BASE / "node_sb3.py")
    control.rpc("h100", "status", dict(role="server"))
    assert calls[-1][1] == str(control.BASE / "node.py")


def test_another_chains_port_is_refused_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(node, "BASE", tmp_path)
    monkeypatch.setattr(node, "session_exists", lambda receipt: False)
    monkeypatch.setattr(node, "port_busy", lambda port: port == 23210)
    monkeypatch.setattr(node, "command", lambda *a, **k: pytest.fail("foreign port must not launch"))
    spec = dict(role="server", out=str(tmp_path / "runs/B/server_23210"), port=23210, tag="B_23210")
    with pytest.raises(RuntimeError, match="occupied port 23210"):
        node.launch(spec)
    assert not list((tmp_path / "runs").rglob("*.owner.json"))


@pytest.mark.parametrize("total,accepted", [(64, True), (65, False)])
def test_new_island_node_enforces_64_worker_cap(tmp_path, monkeypatch, total, accepted):
    monkeypatch.setattr(node, "ISLAND", tmp_path)
    monkeypatch.setattr(node, "T107_ISLAND", tmp_path)
    monkeypatch.setattr(node, "session_exists", lambda receipt: False)
    monkeypatch.setattr(node, "wait_workers", lambda: None)
    monkeypatch.setattr(node, "call", lambda *a, **k: "")
    spec = dict(role="driver", out=str(tmp_path / "os_cl/runs/B"), arm="B", suite="libero_10",
                servers=["host:23220"], workers=[total], env={})
    if accepted:
        node.launch(spec)
        assert (tmp_path / "os_cl/runs/B/driver.job.json").exists()
        job = json.loads((tmp_path / "os_cl/runs/B/driver.job.json").read_text())
        assert job["env"]["WORKER_HOST"] == "timan107"
    else:
        with pytest.raises(ValueError, match="1..64"):
            node.launch(spec)


def test_t107_node_refuses_payload_that_would_select_old_island(tmp_path, monkeypatch):
    monkeypatch.setattr(node, "ISLAND", tmp_path)
    monkeypatch.setattr(node, "T107_ISLAND", tmp_path)
    monkeypatch.setattr(node, "session_exists", lambda receipt: False)
    monkeypatch.setattr(node, "wait_workers", lambda: pytest.fail("must refuse before launching"))
    spec = dict(role="driver", out=str(tmp_path / "os_cl/runs/B"), arm="B", suite="libero_10",
                servers=["host:23220"], workers=[1], env={"WORKER_HOST": "timan108"})
    with pytest.raises(ValueError, match="new timan107 island requires"):
        node.launch(spec)


@pytest.mark.parametrize("action", ["start", "start-server", "status", "stop", "install", "pull", "archive", "checkpoints", "manifest", "supervise", "--help"])
def test_old_rpc_cli_is_byte_compatible_with_frozen_sb2(tmp_path, monkeypatch, capsys, action):
    """Replay legacy payloads through both CLIs; compare output/calls/files."""
    import types
    old = types.ModuleType("node_sb2")
    old.__file__ = node.__file__
    baseline = Path(node.__file__).with_name("fixtures") / "node_sb2.txt"
    exec(compile(baseline.read_text(), str(baseline), "exec"), old.__dict__)
    assert old.ISLAND == node.ISLAND
    root = tmp_path / "tree"
    root.mkdir()
    (root / "code.py").write_text("code")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("cfg")
    payloads = {
        "start": dict(role="driver", out=str(tmp_path / "os_cl/runs/arm"), arm="arm", suite="libero_spatial",
                      servers=["host:23210"], workers=[40], env={}),
        "start-server": dict(role="server", out=str(tmp_path / "runs/arm/server_23210"), port=23210, tag="arm_23210",
                             model="pi05", suite="libero_spatial", yaml=str(cfg), plugin=[], env={}, need_mb=3000),
        "status": dict(role="driver", out=str(tmp_path / "missing")),
        "stop": dict(role="driver", arm="arm", out=str(tmp_path / "missing")),
        "install": dict(dest=str(tmp_path / "install"), archive=str(tmp_path / "input.tar"), hashes={"cfg": node.sha(cfg)}),
        "pull": dict(dest=str(tmp_path / "assets"), files=[dict(rel="cfg", size=3, sha256=node.sha(cfg))], url="rsync://unused"),
        "archive": dict(root=str(root), archive=str(tmp_path / "logs.tar"), prefix="client"),
        "checkpoints": {"rule": "unchanged"},
        "manifest": dict(root=str(root)),
        "supervise": dict(log=str(tmp_path / "job.log"), cmd=["test-child"], env={}, cwd=str(root),
                          pidfile=str(tmp_path / "job.pid"), receipt=str(tmp_path / "job.owner.json"), role="driver"),
        "--help": {},
    }
    import tarfile
    with tarfile.open(tmp_path / "input.tar", "w") as tf:
        tf.add(cfg, arcname="cfg")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(payloads[action]))
    outcomes = []
    for module in (old, node):
        calls = []
        monkeypatch.setattr(module, "BASE", tmp_path)
        monkeypatch.setattr(module, "TREE", root)
        monkeypatch.setattr(module, "ISLAND", tmp_path)
        monkeypatch.setattr(module, "session_exists", lambda receipt: False)
        monkeypatch.setattr(module, "wait_workers", lambda: None)
        monkeypatch.setattr(module, "wait_fleet", lambda: None)
        monkeypatch.setattr(module, "port_busy", lambda port: False)
        def call(argv, **kw):
            calls.append((list(map(str, argv)), kw))
            if argv[0] == "rsync":
                (tmp_path / "assets/cfg").write_text("cfg")
            return "81000" if argv[0] == "nvidia-smi" else ""
        monkeypatch.setattr(module, "call", call)
        monkeypatch.setattr(module.os, "statvfs", lambda p: SimpleNamespace(f_bavail=100 << 30, f_frsize=1))
        monkeypatch.setattr(module.time, "time_ns", lambda: 123)
        if action == "supervise":
            (tmp_path / "job.owner.json").write_text(json.dumps(dict(spec={})))
            monkeypatch.setattr(module, "identity", lambda pid: "starttime")
            child = SimpleNamespace(pid=123, wait=lambda: 0)
            monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: child)
            monkeypatch.setattr(module.signal, "signal", lambda *a: None)
        monkeypatch.setattr("sys.argv", ["node.py", "start" if action == "start-server" else action, str(spec)])
        if action == "--help":
            with pytest.raises(SystemExit) as exc:
                module.main()
            assert exc.value.code == 0
        else:
            module.main()
        files = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        outcomes.append((capsys.readouterr().out, calls, files))
        # Only these test-created outputs need resetting for an identical replay.
        for name in ("os_cl", "assets", "install", "runs"):
            folder = tmp_path / name
            for p in sorted(folder.rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
            if folder.exists():
                folder.rmdir()
        for name in ("logs.tar", "logs.tar.lock", "locks", "job.log", "job.pid", "job.owner.json"):
            p = tmp_path / name
            if p.is_dir():
                for child in p.iterdir():
                    child.unlink()
                p.rmdir()
            elif p.exists():
                p.unlink()
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("cmd", [
    ["tether", "exec", "h100", "--", "nvidia-smi"],
    ["tether", "exec", "h100", "--", "python3", "node.py", "start", "spec.json"],
    ["tether", "exec", "h100", "--", "python3", "node.py", "stop", "spec.json"],
    ["tether", "exec", "timan108", "--", "python3", "purge_exc.py", "out"],
    ["tether", "exec", "timan108", "--", "python3", "count.py", "journal"],
    ["tether", "push", "--force", "local", "h100:remote"],
    ["tether", "pull", "--force", "h100:remote", "local"],
])
@pytest.mark.parametrize("failure", [69, 70, 75, 255, "timeout", "timeout_text"])
def test_tether_retries_idempotent_commands(monkeypatch, cmd, failure):
    calls, sleeps = [], []
    def invoke(argv, **kw):
        calls.append(argv)
        if len(calls) <= 2:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, kw["timeout"])
            return SimpleNamespace(returncode=1 if failure == "timeout_text" else failure,
                                   stdout="", stderr="timed out after 10m" if failure == "timeout_text" else "transport unavailable")
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
    monkeypatch.setattr(control.subprocess, "run", invoke)
    monkeypatch.setattr(control.time, "sleep", sleeps.append)
    assert control.run(cmd) == "ok"
    assert calls == [cmd, cmd, cmd]
    assert sleeps == [2, 4]


@pytest.mark.parametrize("cmd,code", [(["tether", "exec", "h100"], 77), (["python3", "local.py"], 75)])
def test_nontransient_or_local_errors_do_not_retry(monkeypatch, cmd, code):
    calls = []
    def invoke(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=code, stdout="", stderr="permission denied")
    monkeypatch.setattr(control.subprocess, "run", invoke)
    monkeypatch.setattr(control.time, "sleep", lambda _: pytest.fail("must not back off"))
    with pytest.raises(control.CommandError):
        control.run(cmd)
    assert calls == [cmd]


def test_tether_timeout_retry_is_bounded(monkeypatch):
    calls = []
    def invoke(argv, **kw):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, kw["timeout"])
    monkeypatch.setattr(control.subprocess, "run", invoke)
    monkeypatch.setattr(control.time, "sleep", lambda _: None)
    with pytest.raises(subprocess.TimeoutExpired):
        control.run(["tether", "exec", "h100"], timeout=60)
    assert len(calls) == 4


@pytest.mark.parametrize("exists", [True, False])
def test_rpc_reuses_content_addressed_remote_spec(monkeypatch, exists):
    monkeypatch.setattr(control, "_RPC_SPECS", set())
    calls, pushed = [], []
    def remote(machine, cmd, timeout=300):
        calls.append((machine, cmd))
        return "SPEC_EXISTS" if cmd[0] == "bash" and exists else "SPEC_MISSING" if cmd[0] == "bash" else "reply"
    def push(machine, local, target):
        pushed.append((machine, target, json.loads(Path(local).read_text())))
    monkeypatch.setattr(control, "remote", remote)
    monkeypatch.setattr(control, "push", push)
    spec = dict(role="server", port=23210)
    assert control.rpc("h100", "status", spec) == "reply"
    assert control.rpc("h100", "status", spec) == "reply"
    assert len(pushed) == (0 if exists else 1)
    assert len(calls) == 3  # one existence check, then one exec per status poll
    assert calls[1][1][-1] == calls[2][1][-1]


def test_timan_push_merges_finalize_and_tolerates_duplicate_exec(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "T108_STAGE", tmp_path / "stage")
    monkeypatch.setattr(control, "_STAGING_READY", set())
    source, dest = tmp_path / "local", tmp_path / "island/cfg/spec.json"
    source.write_bytes(b"spec")
    commands = []
    def remote(machine, cmd, timeout=300):
        assert machine == "timan108"
        commands.append(cmd)
        # Execute only local mkdir/finalize commands twice, modeling h100's quirk.
        for _ in range(2):
            subprocess.run(list(map(str, cmd)), check=True, capture_output=True)
        return ""
    def run(cmd, **kw):
        assert cmd[:3] == ["tether", "push", "--force"]
        Path(cmd[-1].split(":", 1)[1]).write_bytes(Path(cmd[-2]).read_bytes())
        return ""
    monkeypatch.setattr(control, "remote", remote)
    monkeypatch.setattr(control, "run", run)
    control.push("timan108", source, dest)
    control.push("timan108", source, dest)
    assert dest.read_bytes() == b"spec"
    assert len(commands) == 3  # staging mkdir once, one finalize exec per transfer
    assert all("mkdir -p" in c[-1] and "cp " in c[-1] and "rm -f" in c[-1] for c in commands[1:])
    assert not list((tmp_path / "stage").iterdir())


def mock_chain(tmp_path, monkeypatch, names=("first",), polls=None, counts=None, collect_error=None):
    root = tmp_path / "runs/r08"
    root.mkdir(parents=True)
    rows = [dict(arm=arm, model="pi05", suite="libero_spatial", mode="stock", yaml="cfg.yaml") for arm in names]
    (root / "arms.json").write_text(json.dumps(rows))
    (root / "h100_sync").mkdir()
    (root / "h100_sync/synced.json").write_text(json.dumps(dict(arms=rows, arms_sha256=node.sha(root / "arms.json"))))
    monkeypatch.setattr(control, "CHAIN_LOCK", tmp_path / "runs/h100_chain.lock")
    monkeypatch.setenv("PORTS", "23210")
    monkeypatch.setenv("WPS", "1")
    monkeypatch.setenv("MAX_ATTEMPTS", "3")
    monkeypatch.setenv("OSCL_TASKS", "0,1")
    monkeypatch.setenv("OSCL_EPISODES", "0,1")
    for key in ("OSCL_MANIFEST", "POLL_SECONDS", "STATUS_FAILURE_LIMIT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(control.signal, "signal", lambda *args: None)
    events, sleeps = [], []
    monkeypatch.setattr(control.time, "sleep", sleeps.append)
    counts = iter(counts or ["4 2 4"] * len(names))
    state = dict(active=False, polls=0)
    def rpc(machine, action, spec, **kw):
        events.append((action, spec.get("role"), state["polls"]))
        if action == "manifest":
            return json.dumps(dict(root=str(control.BASE / "openpi"), sha256="a" * 64, files=2, bytes=3))
        if action in ("start", "stop"):
            if spec["role"] == "driver":
                state["active"] = action == "start"
            return action.upper()
        assert kw["timeout"] == 60 and kw["attempts"] == 1
        running = spec["role"] == "server"
        if state["active"]:
            poll = polls[state["polls"]] if polls and state["polls"] < len(polls) else (True, False)
            value = poll[0 if spec["role"] == "server" else 1]
            if spec["role"] == "driver":
                state["polls"] += 1
            if isinstance(value, Exception):
                raise value
            running = value
        return json.dumps(dict(running=running, listening=True, dead=False))
    def remote(machine, cmd, **kw):
        if machine == "h100":
            return "81000"
        if "purge_exc.py" in str(cmd):
            events.append(("purge", None, state["polls"]))
            assert not state["active"], "purge must follow confirmed driver stop"
            return "0"
        events.append(("count", None, state["polls"]))
        assert not state["active"], "count must follow confirmed driver stop"
        return next(counts)
    def collect(runroot, arm, manifest):
        events.append(("collect", arm, state["polls"]))
        if collect_error and arm == names[0]:
            raise collect_error
        return dict(complete=4)
    monkeypatch.setattr(control, "rpc", rpc)
    monkeypatch.setattr(control, "remote", remote)
    monkeypatch.setattr(control, "collect", collect)
    return root, events, sleeps


@pytest.mark.parametrize("role", ["server", "driver"])
def test_failed_poll_is_unknown_and_recovery_resets_counter(tmp_path, monkeypatch, role):
    failure = subprocess.TimeoutExpired("tether", 60)
    unknown = (failure, True) if role == "server" else (True, failure)
    polls = [unknown] * 9 + [(True, True)] + [unknown] * 9 + [(True, False)]
    root, events, sleeps = mock_chain(tmp_path, monkeypatch, polls=polls)
    control.chain(root, ["first"])
    assert (root / "state/first.DONE").exists()
    assert sleeps == [60] * 20
    assert [(e[1], e[2]) for e in events if e[0] == "stop"] == [("driver", 20), ("server", 20)]
    assert "SERVER_DIED" not in (root / "runs/chain.log").read_text()
    assert "STATUS_UNKNOWN" in (root / "runs/chain.log").read_text()


@pytest.mark.parametrize("role", ["server", "driver"])
def test_ten_consecutive_failed_polls_stop_with_error(tmp_path, monkeypatch, role):
    failure = RuntimeError("too_many_in_flight")
    unknown = (failure, True) if role == "server" else (True, failure)
    root, events, sleeps = mock_chain(tmp_path, monkeypatch, polls=[unknown] * 10)
    with pytest.raises(RuntimeError, match="STATUS_POLL_FAILED consecutive=10"):
        control.chain(root, ["first"])
    assert sleeps == [60] * 10
    assert all(e[2] == 10 for e in events if e[0] == "stop")
    assert not any(e[0] in ("purge", "count", "collect") for e in events)
    assert (root / "state/CHAIN.ERROR").exists()
    assert not (root / "state/first.DONE").exists()


def test_boot_status_failure_does_not_abort_immediately(tmp_path, monkeypatch):
    root, events, sleeps = mock_chain(tmp_path, monkeypatch)
    rpc = control.rpc
    failures = [True, True]
    def boot_rpc(machine, action, spec, **kw):
        if action == "status" and not any(e[0] == "start" and e[1] == "driver" for e in events) and failures:
            failures.pop()
            raise subprocess.TimeoutExpired("tether", 60)
        return rpc(machine, action, spec, **kw)
    monkeypatch.setattr(control, "rpc", boot_rpc)
    control.chain(root, ["first"])
    assert sleeps == [5, 5, 60]
    assert (root / "state/first.DONE").exists()


@pytest.mark.parametrize("error", [RuntimeError("SHA mismatch"), subprocess.TimeoutExpired("tether pull", 590)])
def test_collection_pending_continues_to_next_arm(tmp_path, monkeypatch, error):
    root, events, _ = mock_chain(tmp_path, monkeypatch, names=("first", "second"), collect_error=error)
    control.chain(root, ["first", "second"])
    assert not (root / "state/first.DONE").exists()
    assert "COLLECT_PENDING" in (root / "state/first.ERROR").read_text()
    assert (root / "state/second.DONE").exists()
    assert not (root / "state/CHAIN.DONE").exists()
    assert "COLLECT_PENDING: first" in (root / "state/CHAIN.ERROR").read_text()
    assert (root / "state/current").read_text() == "none\n"
    assert [e[1] for e in events if e[0] == "collect"] == ["first", "second"]
    assert "COLLECT_PENDING arm=first" in (root / "runs/chain.log").read_text()


def test_collect_pull_retries_timeout_before_sha_and_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "rpc", lambda *a, **k: '{"sha256":"irrelevant"}')
    calls, sleeps = [], []
    def run(cmd, **kw):
        calls.append(cmd)
        assert kw["attempts"] == 1  # collect owns its eight-attempt retry budget
        if len(calls) < 3:
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        raise ValueError("reached third pull")
    monkeypatch.setattr(control, "run", run)
    monkeypatch.setattr(control.time, "sleep", sleeps.append)
    with pytest.raises(ValueError, match="reached third pull"):
        control.collect(tmp_path, "sample")
    assert len(calls) == 3 and sleeps == [15, 30]


def test_collect_pull_timeout_exhausts_eight_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "rpc", lambda *a, **k: '{"sha256":"irrelevant"}')
    calls, sleeps = [], []
    def run(cmd, **kw):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(control, "run", run)
    monkeypatch.setattr(control.time, "sleep", sleeps.append)
    with pytest.raises(subprocess.TimeoutExpired):
        control.collect(tmp_path, "sample")
    assert len(calls) == 8 and sleeps == [15, 30, 45, 60, 60, 60, 60]
    assert not (tmp_path / "runs/sample").exists()


def test_collection_completion_recheck_still_required(tmp_path, monkeypatch):
    root, _, _ = mock_chain(tmp_path, monkeypatch)
    monkeypatch.setattr(control, "collect", lambda *a: dict(complete=3))
    control.chain(root, ["first"])
    assert not (root / "state/first.DONE").exists()
    assert "completion count changed" in (root / "state/first.ERROR").read_text()


def test_all_requested_arms_checked_before_first_remote_call(tmp_path, monkeypatch):
    root, events, _ = mock_chain(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="missing from synced.json: missing"):
        control.chain(root, ["first", "missing"])
    assert events == []


def test_code_tree_digest_recorded_once_for_every_arm(tmp_path, monkeypatch):
    root, events, _ = mock_chain(tmp_path, monkeypatch, names=("first", "second"))
    control.chain(root, ["first", "second"])
    assert sum(e[0] == "manifest" for e in events) == 1
    for arm in ("first", "second"):
        launch = json.loads((root / f"runs/{arm}/h100_launch.json").read_text())
        assert launch["code_tree"]["sha256"] == "a" * 64
        assert "launch_id" in launch["driver"] and "launch_id" in launch["servers"][0]


def test_remote_code_manifest_tracks_content_and_ignores_bytecode(tmp_path, monkeypatch):
    monkeypatch.setattr(node, "TREE", tmp_path)
    (tmp_path / "a.py").write_bytes(b"print(1)\n")
    (tmp_path / "cfg.yaml").write_bytes(b"x: 1\n")
    first = node.code_manifest(dict(root=str(tmp_path)))
    record = "a.py\0" + node.sha(tmp_path / "a.py") + "\ncfg.yaml\0" + node.sha(tmp_path / "cfg.yaml") + "\n"
    assert first["sha256"] == hashlib.sha256(record.encode()).hexdigest()
    assert first["files"] == 2 and first["bytes"] == 14
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/a.pyc").write_bytes(b"generated")
    assert node.code_manifest(dict(root=str(tmp_path)))["sha256"] == first["sha256"]
    (tmp_path / "a.py").write_bytes(b"print(2)\n")
    assert node.code_manifest(dict(root=str(tmp_path)))["sha256"] != first["sha256"]
    with pytest.raises(ValueError, match="isolated"):
        node.code_manifest(dict(root=str(tmp_path / "elsewhere")))


@pytest.mark.parametrize("action", ["setup", "sync"])
@pytest.mark.parametrize("legacy", [False, True])
def test_maintenance_refuses_active_global_or_legacy_chain(tmp_path, monkeypatch, action, legacy):
    root = tmp_path / "runs/r08"
    lockpath = root / "state/h100_chain.lock" if legacy else tmp_path / "runs/h100_chain.lock"
    lockpath.parent.mkdir(parents=True)
    monkeypatch.setattr(control, "CHAIN_LOCK", tmp_path / "runs/h100_chain.lock")
    monkeypatch.setattr(control, "_setup", lambda: pytest.fail("setup must not mutate"))
    monkeypatch.setattr(control, "_sync", lambda *a, **k: pytest.fail("sync must not mutate"))
    with open(lockpath, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="H100_CHAIN_BUSY"):
            control.setup() if action == "setup" else control.sync(root, ["first"])
    # A refused partial acquisition must have released the global lock.
    with control.chain_lock(root):
        pass


def test_chain_maintenance_serialization_and_plan_remains_read_only(tmp_path, monkeypatch):
    root, events, _ = mock_chain(tmp_path, monkeypatch)
    monkeypatch.setattr(control, "_sync", lambda *a, **k: "plan")
    with control.chain_lock(root):
        with pytest.raises(RuntimeError, match="H100_CHAIN_BUSY"):
            control.chain(root, ["first"])
        assert control.sync(root, ["first"], plan_only=True) == "plan"
    assert events == []


def test_chain_reads_synced_plan_under_maintenance_lock(tmp_path, monkeypatch):
    root, _, _ = mock_chain(tmp_path, monkeypatch)
    sha = control.sha
    def guarded_sha(path):
        with pytest.raises(RuntimeError, match="H100_CHAIN_BUSY"):
            with control.chain_lock(root):
                pass
        return sha(path)
    monkeypatch.setattr(control, "sha", guarded_sha)
    control.chain(root, ["first"])
    assert (root / "state/first.DONE").exists()


def test_island_workers_ignore_other_users_and_other_trees(tmp_path, monkeypatch):
    monkeypatch.setattr(node, "ISLAND", tmp_path / "island")
    entries = [
        (1, os.getuid(), str(node.ISLAND / "pool")),
        (2, os.getuid() + 1, str(node.ISLAND / "pool")),
        (3, os.getuid(), str(tmp_path / "other/pool")),
        (4, os.getuid(), str(tmp_path / "island_sibling/pool")),
    ]
    uids = {}
    for pid, uid, pool in entries:
        path = tmp_path / f"proc/{pid}/cmdline"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"python\0-m\0examples.libero.worker_entry\0--init-states-dir\0" + pool.encode() + b"\0")
        uids[path] = uid
    stat = Path.stat
    def fake_stat(path, **kw):
        return SimpleNamespace(st_uid=uids[path]) if path in uids else stat(path, **kw)
    monkeypatch.setattr(Path, "stat", fake_stat)
    assert node.island_workers(tmp_path / "proc") == 1


def fake_clock(monkeypatch):
    clock = SimpleNamespace(now=0, sleeps=[])
    monkeypatch.setattr(node.time, "monotonic", lambda: clock.now)
    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds
    monkeypatch.setattr(node.time, "sleep", sleep)
    return clock


def test_worker_busy_wait_recovers_and_has_120_second_bound(monkeypatch):
    clock = fake_clock(monkeypatch)
    counts = iter([2, 1, 0])
    monkeypatch.setattr(node, "island_workers", lambda: next(counts))
    node.wait_workers()
    assert clock.sleeps == [2, 2]
    monkeypatch.setattr(node, "island_workers", lambda: 1)
    start = clock.now
    with pytest.raises(RuntimeError, match="WORKERS_BUSY existing=1"):
        node.wait_workers()
    assert clock.now - start == 120


def test_driver_launch_supervises_bash_without_outer_flock(tmp_path, monkeypatch):
    monkeypatch.setattr(node, "ISLAND", tmp_path)
    monkeypatch.setattr(node, "session_exists", lambda receipt: False)
    monkeypatch.setattr(node, "wait_workers", lambda: None)
    monkeypatch.setattr(node, "call", lambda *a, **k: "")
    out = tmp_path / "os_cl/runs/sample"
    node.launch(dict(role="driver", out=str(out), arm="sample", suite="libero_spatial", servers=["host:23210"], workers=[1], env={}))
    job = json.loads((out / "driver.job.json").read_text())
    assert job["cmd"][:2] == ["bash", str(tmp_path / "os_cl/run_arm.sh")]


def test_run_arm_lock_refuses_before_any_driver_work(tmp_path):
    source = Path(node.__file__).with_name("run_arm.sh").read_text()
    script = tmp_path / "run_arm.sh"
    script.write_text(source.replace("R=/scratch/zixuans8/openpi_trace", f"R={tmp_path}"))
    lockpath = tmp_path / "os_cl/locks/fleet.lock"
    lockpath.parent.mkdir(parents=True)
    with open(lockpath, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", str(script), "suite", "arm", "host:port", "1", "out"], capture_output=True)
    assert result.returncode == 75 and result.stdout == b"" and result.stderr == b""


def test_legacy_driver_stop_waits_for_fleet_lock_release(tmp_path, monkeypatch):
    monkeypatch.setattr(node, "ISLAND", tmp_path)
    monkeypatch.setattr(node, "ours", lambda receipt: False)
    monkeypatch.setattr(node, "session_exists", lambda receipt: False)
    lockpath = tmp_path / "os_cl/locks/fleet.lock"
    lockpath.parent.mkdir(parents=True)
    out = tmp_path / "os_cl/runs/arm"
    out.mkdir(parents=True)
    spec = dict(role="driver", out=str(out), arm="arm")
    (out / "driver.owner.json").write_text(json.dumps(dict(spec=spec, session="oscl_arm", socket=[])))
    checks = []
    monkeypatch.setattr(node, "wait_workers", lambda: checks.append("workers exited"))
    with open(lockpath, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def release(seconds):
            assert not json.loads((out / "driver.owner.json").read_text()).get("stopped")
            checks.append("waited for real driver")
            fcntl.flock(held, fcntl.LOCK_UN)
        monkeypatch.setattr(node.time, "sleep", release)
        node.stop(spec)
    assert checks == ["waited for real driver", "workers exited"]
    assert json.loads((out / "driver.owner.json").read_text())["stopped"]


def test_fleet_stop_wait_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(node, "ISLAND", tmp_path)
    clock = fake_clock(monkeypatch)
    lockpath = tmp_path / "os_cl/locks/fleet.lock"
    lockpath.parent.mkdir(parents=True)
    with open(lockpath, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="FLEET_STOP_TIMEOUT"):
            node.wait_fleet()
    assert clock.now == 180


def test_missing_driver_receipt_still_requires_fleet_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(node, "ISLAND", tmp_path)
    monkeypatch.setattr(node, "wait_fleet", lambda: (_ for _ in ()).throw(RuntimeError("FLEET_STOP_TIMEOUT")))
    monkeypatch.setattr(node, "wait_workers", lambda: pytest.fail("fleet has not exited yet"))
    with pytest.raises(RuntimeError, match="FLEET_STOP_TIMEOUT"):
        node.stop(dict(role="driver", out=str(tmp_path / "os_cl/runs/missing"), arm="missing"))


@pytest.mark.parametrize("recycled", [False, True])
def test_server_stop_escalates_only_recorded_owned_pid(tmp_path, monkeypatch, recycled):
    monkeypatch.setattr(node, "BASE", tmp_path)
    monkeypatch.setattr(node, "ours", lambda receipt: False)
    monkeypatch.setattr(node, "session_exists", lambda receipt: False)
    fake_clock(monkeypatch)
    out = tmp_path / "runs/arm"
    out.mkdir(parents=True)
    spec = dict(role="server", out=str(out), tag="arm", port=23210, yaml="owned-config.yaml")
    (out / "server_arm.owner.json").write_text(json.dumps(dict(spec=spec, session="oscl23210", socket=[])))
    (out / "server_arm.pid").write_text(json.dumps(dict(pid=99999999, starttime="owned")))
    state, signals = dict(identity="owned"), []
    monkeypatch.setattr(node, "identity", lambda pid: state["identity"])
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: b"python\0serve.py\0owned-config.yaml\0" if str(p) == "/proc/99999999/cmdline" else read_bytes(p))
    def term(pid, sig):
        assert pid == 99999999
        signals.append(sig)
        if recycled:
            state["identity"] = "foreign"
    def kill(pid, sig):
        assert pid == 99999999 and state["identity"] == "owned"
        signals.append(sig)
        state["identity"] = None
    monkeypatch.setattr(node.os, "killpg", term)
    monkeypatch.setattr(node.os, "kill", kill)
    node.stop(spec, timeout_s=2)
    assert signals == ([signal.SIGTERM] if recycled else [signal.SIGTERM, signal.SIGKILL])
    assert json.loads((out / "server_arm.owner.json").read_text())["stopped"]
