"""h100/timan108 harness. Existing scripts and collector behavior are unchanged."""
import argparse
import ast
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tarfile
import tempfile
import time
import uuid

from .assets import BASE, ISLAND, build_plan, remap
from .node import atomic, identity, sha
from .fleet import FLEETS, island, worker_host

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REMOTE_NODE = {"h100": str(BASE / "node.py"), "timan108": str(ISLAND / "os_cl/node.py")}
REMOTE_NODE["timan107"] = str(island("timan107") / "os_cl/node.py")
PY = REPO / ".venv/bin/python"
T108_STAGE = Path("/srv/local/zixuans8/oscl_sb_stage")  # md0, same disk as /scratch; tether allows /srv
T107_STAGE = Path("/tmp/oscl_sb3_stage")  # tether allow_roots excludes /scratch on timan107
CHAIN_LOCK = Path("/home/weiland/trace_runs/os_closed_loop/h100_chain.lock")
POLL_SECONDS = 60
STATUS_FAILURE_LIMIT = 10
_RPC_SPECS = set()
_STAGING_READY = set()


class CommandError(RuntimeError):
    def __init__(self, cmd, result):
        super().__init__(f"command failed ({result.returncode}): {shlex.join(cmd)}\n{result.stdout}\n{result.stderr}")
        self.transient = result.returncode in (69, 70, 75, 255) or any(
            word in (result.stdout + result.stderr).lower()
            for word in ("too_many_in_flight", "timed out", "timeout", "deadline exceeded"))


def run(cmd, timeout=300, attempts=None, **kw):
    """Retry only tether calls: all callers use idempotent remote operations."""
    cmd = list(map(str, cmd))
    tether = Path(cmd[0]).name == "tether"
    tries = attempts if attempts is not None else (4 if tether else 1)
    if tries < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(tries):
        try:
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, **kw)
            if p.returncode:
                raise CommandError(cmd, p)
            return p.stdout.strip()
        except (CommandError, subprocess.TimeoutExpired) as exc:
            if not tether or attempt == tries - 1 or isinstance(exc, CommandError) and not exc.transient:
                raise
            delay = min(2 ** (attempt + 1), 30)
            print(f"TETHER_RETRY attempt={attempt+1}/{tries} delay={delay}s error={str(exc)[:300]}", flush=True)
            time.sleep(delay)


def remote(node, cmd, timeout=300, attempts=None):
    return run(["tether", "exec", node, "--", "bash", "-c", shlex.join(list(map(str, cmd)))],
               timeout=timeout, attempts=attempts)


def push(node, local, destination):
    if node in FLEETS:
        stage_root = T108_STAGE if node == "timan108" else T107_STAGE
        if node not in _STAGING_READY:
            remote(node, ["mkdir", "-p", stage_root])
            _STAGING_READY.add(node)
        stage = stage_root / (sha(local) + ".push")
        run(["tether", "push", "--force", local, f"{node}:{stage}"])
        # One idempotent exec for mkdir/cp/rm, including duplicate execution after rm.
        qstage, qdest = shlex.quote(str(stage)), shlex.quote(str(destination))
        script = (f"mkdir -p {shlex.quote(str(Path(destination).parent))} && "
                  f"if test -f {qstage}; then cp {qstage} {qdest} && rm -f {qstage}; fi && "
                  f"test \"$(sha256sum {qdest} | cut -d ' ' -f 1)\" = {sha(local)}")
        remote(node, ["bash", "-c", script])
    else:
        run(["tether", "push", "--force", local, f"{node}:{destination}"])


def rpc(node, action, spec, timeout=300, attempts=None):
    payload = json.dumps(spec, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    root = BASE / ".rpc" if node == "h100" else island(node) / "os_cl/.rpc"
    target = str(root / (digest + ".json"))
    key = (node, target)
    if key not in _RPC_SPECS:
        # A fresh controller also reuses specs already uploaded by an earlier one.
        script = (f"if test -f {shlex.quote(target)} && "
                  f"test \"$(sha256sum {shlex.quote(target)} | cut -d ' ' -f 1)\" = {digest}; then echo SPEC_EXISTS; "
                  f"else mkdir -p {shlex.quote(str(root))} && echo SPEC_MISSING; fi")
        if remote(node, ["bash", "-c", script]).splitlines()[-1] != "SPEC_EXISTS":
            with tempfile.NamedTemporaryFile() as f:
                f.write(payload)
                f.flush()
                push(node, f.name, target)
        _RPC_SPECS.add(key)
    # Concurrent sync is the only new h100 RPC. Its separately deployed helper
    # leaves A's executable helper and application snapshot completely intact.
    helper = str(BASE / "node_sb3.py") if node == "h100" and action == "pull-new" else REMOTE_NODE[node]
    cmd = ["python3", helper, action, target]
    out = remote(node, cmd, timeout) if attempts is None else remote(node, cmd, timeout, attempts=attempts)
    return out


def json_output(out):
    # h100 sometimes executes twice; use the last complete one-line JSON reply.
    for line in reversed(out.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass
    raise ValueError("no JSON reply: " + out[-1000:])


def bundle(node, destination, paths):
    with tempfile.TemporaryDirectory(prefix="oscl_sb_bundle_") as td:
        archive = Path(td) / "code.tar"
        hashes = {}
        with tarfile.open(archive, "w") as tf:
            for local, relative in paths:
                hashes[str(relative)] = sha(local)
                tf.add(local, arcname=str(relative), recursive=False)
        stage = (BASE if node == "h100" else island(node) / "os_cl") / ("bundle_" + sha(archive)[:20] + ".tar")
        push(node, archive, stage)
        print(rpc(node, "install", dict(dest=str(destination), archive=str(stage), hashes=hashes)))
        # Only the content-addressed transfer archive we just created.
        remote(node, ["rm", "-f", stage])


def source_files():
    result = {}
    roots = ("src", "scripts", "assets", "packages/openpi-client/src", "exp/offline_search", "exp/libero_groot",
             "exp/gate_threshold_pareto", "exp/ablation_study/cache_size", "examples/libero")
    for root in roots:
        for p in (REPO / root).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".yaml", ".json") and "__pycache__" not in p.parts:
                # Results are research artifacts, not executable source dependencies.
                if root == "exp/offline_search" and p.suffix != ".py":
                    continue
                result[p.relative_to(REPO)] = p
    # Follow static absolute experiment imports, rather than copying unrelated research trees.
    pending = list(result.values())
    seen = set()
    while pending:
        p = pending.pop()
        if p in seen or p.suffix != ".py":
            continue
        seen.add(p)
        for n in ast.walk(ast.parse(p.read_text())):
            mods = [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module else []
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            for mod in mods:
                if mod.startswith("exp."):
                    target = REPO / mod.replace(".", "/")
                    candidates = [target.with_suffix(".py")] if target.with_suffix(".py").exists() else list(target.glob("*.py"))
                    for q in candidates:
                        if q.relative_to(REPO) not in result:
                            result[q.relative_to(REPO)] = q
                            pending.append(q)
    for suite in ("libero_spatial", "libero_10"):
        for p in (REPO / f"exp/common/data/db_init/libero/{suite}_apool").rglob("*"):
            if p.is_file():
                result[p.relative_to(REPO)] = p
    return [(p, rel) for rel, p in sorted(result.items())]


@contextmanager
def chain_lock(root=None):
    """Full maintenance: exclusive gate, global M5 lock, hosts and legacy roots."""
    CHAIN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    legacy = set(CHAIN_LOCK.parent.glob("*/state/h100_chain.lock"))
    if root is not None:
        legacy.add(Path(root).resolve() / "state/h100_chain.lock")
    with ExitStack() as stack:
        for path in [Path(str(CHAIN_LOCK) + ".gate"), CHAIN_LOCK,
                     *(host_lock(h) for h in FLEETS), *sorted(legacy)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(open(path, "a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"H100_CHAIN_BUSY lock={path}; stop the owned chain before maintenance/relaunch") from exc
        yield


def host_lock(host):
    return CHAIN_LOCK.parent / ("h100_fleet_" + host + ".lock")


def legacy_chains(ignore_pid=None):
    """Authenticate holders of old M5 locks, including its inactive-root locks.

    Do not infer ownership from a stale receipt. Linux's lock table must identify
    a live controller whose PID/starttime/argv match its own root's receipt.
    Unknown holders fail closed. No process or lock held by A is modified.
    """
    roots = {p.parent.parent for p in CHAIN_LOCK.parent.glob("*/state/h100_chain.lock")}
    for host in FLEETS:
        registry = host_lock(host).with_suffix(".root.json")
        if registry.exists():
            roots.add(Path(json.loads(registry.read_text())["root"]))
    paths = [CHAIN_LOCK, *(root / "state/h100_chain.lock" for root in roots)]
    holders = set()
    table = Path("/proc/locks").read_text().splitlines()
    for path in paths:
        if not path.exists():
            continue
        st = path.stat()
        for line in table:
            fields = line.split()
            if len(fields) < 8 or fields[1] != "FLOCK" or fields[2] != "ADVISORY":
                continue
            major, minor, ino = fields[5].split(":")
            if (int(major, 16), int(minor, 16), int(ino)) == (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino):
                holders.add(int(fields[4]))
    result = []
    for pid in holders:
        if pid == ignore_pid:
            continue
        found = None
        for root_path in roots:
            path = root_path / "state/h100_chain.owner.json"
            if not path.exists():
                continue
            receipt = json.loads(path.read_text())
            root = path.parent.parent.resolve()
            if receipt.get("pid") != pid or identity(pid) != receipt.get("starttime"):
                continue
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            argv = raw.decode().split("\0")
            module = "exp.offline_search.closed_loop.ops.h100.control"
            if raw.hex() != receipt.get("cmdline") or module not in argv:
                continue
            pos = argv.index(module)
            if argv[pos + 1:pos + 3] != ["chain", str(root)]:
                continue
            # Read just the topology selector; never print process environments.
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            host = next((v.split(b"=", 1)[1].decode() for v in env if v.startswith(b"WORKER_HOST=")), "timan108")
            if host not in FLEETS:
                raise RuntimeError(f"H100_CHAIN_BUSY unknown fleet pid={pid}")
            found = dict(root=root, worker_host=host)
            break
        if found is None:
            raise RuntimeError(f"H100_CHAIN_BUSY unauthenticated lock holder pid={pid}")
        result.append(found)
    return result


@contextmanager
def fleet_lock(root=None):
    """Chains/scoped maintenance share the global gate and own one host/root."""
    host = worker_host()
    CHAIN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        paths = [(Path(str(CHAIN_LOCK) + ".gate"), fcntl.LOCK_SH), (host_lock(host), fcntl.LOCK_EX)]
        if root is not None:
            paths.append((Path(root).resolve() / "state/h100_chain.lock", fcntl.LOCK_EX))
        for path, mode in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(open(path, "a"))
            try:
                fcntl.flock(lock, mode | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"H100_CHAIN_BUSY lock={path}") from exc
        if root is not None:
            atomic(host_lock(host).with_suffix(".root.json"), dict(root=str(Path(root).resolve())))
        active = legacy_chains(ignore_pid=os.getpid())
        if any(a["worker_host"] == host or root is not None and a["root"].name == Path(root).name for a in active):
            raise RuntimeError(f"H100_CHAIN_BUSY fleet={host}")
        yield active


def setup(worker_only=False):
    with fleet_lock() if worker_only else chain_lock():
        _setup(worker_only=worker_only) if worker_only else _setup()


def _setup(worker_only=False):
    host, worker_island = worker_host(), island()
    nodes = [(host, worker_island / "os_cl")]
    if not worker_only:
        nodes.insert(0, ("h100", BASE))
    for node, directory in nodes:
        remote(node, ["mkdir", "-p", directory])
        push(node, HERE / "node.py", REMOTE_NODE[node])
    if not worker_only:
        push("h100", HERE / "node.py", BASE / "node_sb3.py")
    files = source_files()
    print(f"CODE_SNAPSHOT files={len(files)} bytes={sum(p.stat().st_size for p, _ in files)}")
    if not worker_only:
        bundle("h100", BASE / "openpi", files)
    worker = files + [(HERE / "run_arm.sh", Path("os_cl/run_arm.sh"))]
    # WorkerAgent intentionally resets PYTHONPATH to island root/src. Its client
    # must win over the sim env's editable install pointing at openpi_dispatch.
    for p, rel in files:
        prefix = Path("packages/openpi-client/src")
        if rel.parts[:len(prefix.parts)] == prefix.parts:
            worker.append((p, Path("src") / rel.relative_to(prefix)))
    for name in ("run_gtp_subset.py", "count.py", "purge_exc.py"):
        worker.append((HERE.parent / "remote" / name, Path("os_cl") / name))
    for name in ("render_test.py", "probe.py", "verify_worker.py"):
        worker.append((HERE / name, Path("os_cl") / name))
    # run_gtp_subset inserts the canonical island path: keep that path on t108.
    with tempfile.TemporaryDirectory(prefix="oscl_sb_cfg_") as td:
        import yaml
        for suite in ("libero_spatial", "libero_10"):
            path = Path(td) / ("apool_" + suite + ".yaml")
            data = yaml.safe_load((REPO / f"exp/ablation_study/cache_size/config/apool_{suite}.yaml").read_text())
            data["apool_dir"] = str(worker_island / f"exp/common/data/db_init/libero/{suite}_apool")
            path.write_text(yaml.safe_dump(data, sort_keys=False))
            worker = [(p, r) for p, r in worker if str(r) != f"exp/ablation_study/cache_size/config/apool_{suite}.yaml"]
            worker.append((path, Path(f"exp/ablation_study/cache_size/config/apool_{suite}.yaml")))
        # Own config under scratch; never update the user's ~/.libero or sim environment.
        cfg = dict(assets="/home/zixuans8/.cache/libero/assets",
                   bddl_files="/scratch/zixuans8/libero_sim/lib/python3.8/site-packages/libero/libero/bddl_files",
                   benchmark_root="/scratch/zixuans8/libero_sim/lib/python3.8/site-packages/libero/libero",
                   datasets="/scratch/zixuans8/libero_sim/lib/python3.8/site-packages/libero/datasets",
                   init_states="/scratch/zixuans8/libero_sim/lib/python3.8/site-packages/libero/libero/init_files")
        path = Path(td) / "config.yaml"
        path.write_text(yaml.safe_dump(cfg))
        worker.append((path, Path("os_cl/libero/config.yaml")))
        # The unchanged legacy subset helper inserts the old island explicitly.
        # Relocate that one copied helper only, never edit the old island/source.
        if host == "timan107":
            subset = Path(td) / "run_gtp_subset.py"
            subset.write_text((HERE.parent / "remote/run_gtp_subset.py").read_text().replace(str(ISLAND), str(worker_island)))
            worker = [(p, r) for p, r in worker if str(r) != "os_cl/run_gtp_subset.py"]
            worker.append((subset, Path("os_cl/run_gtp_subset.py")))
        bundle(host, worker_island, worker)


@contextmanager
def daemon(work, files):
    """Read-only, per-sync rsync daemon. It exposes only selected staged files."""
    work = Path(work)
    staged = tempfile.TemporaryDirectory(prefix="export_", dir=work)
    stage = Path(staged.name)
    for item in files:
        dest = stage / item["rel"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Current rsync hardens daemon symlinks against escaping the module.
        # Hardlinks expose exact selected files without duplicating their bytes.
        # Cross-filesystem inputs (e.g. /data libraries) are copied into the stage, loudly.
        src = Path(item["source"]).resolve()
        if src.stat().st_dev == stage.stat().st_dev:
            os.link(src, dest)
        else:
            import shutil
            print(f"COPY_FALLBACK cross-device {src} ({src.stat().st_size} bytes)", flush=True)
            shutil.copy2(src, dest)
    port = int(os.environ.get("SYNC_PORT", "23197"))
    if not 23100 <= port <= 23199 or port in (23198, 23199):
        raise ValueError("SYNC_PORT must use 23100..23197; 23198/23199 are reserved")
    # Probe bind first and let rsync recheck atomically. Never stop an existing listener.
    import socket
    with socket.socket() as s:
        s.bind(("0.0.0.0", port))
    config = work / "rsyncd.conf"
    # Non-root daemon defaults retain its user/groups; explicit uid/gid tries
    # setgroups(), which is unavailable to this session.
    config.write_text(f"pid file = {work}/rsync.pid\nlog file = {work}/rsync.log\nuse chroot = no\n[oscl_sb]\npath = {stage}\nread only = yes\n")
    proc = subprocess.Popen(["rsync", "--daemon", "--no-detach", "--port", str(port), "--config", str(config)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        time.sleep(.3)
        if proc.poll() is not None:
            raise RuntimeError("rsync daemon failed: " + proc.stderr.read().decode())
        yield f"rsync://{os.environ.get('SYNC_HOST', 'ziyanglin.com')}:{port}/oscl_sb/"
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
        staged.cleanup()


def sync(runroot, names, plan_only=False, concurrent=False):
    if plan_only:
        return _sync(runroot, names, plan_only=True)
    if concurrent:
        with fleet_lock(runroot) as active:
            return _sync(runroot, names, concurrent=True, active=active)
    with chain_lock(runroot):
        return _sync(runroot, names)


def _sync(runroot, names, plan_only=False, concurrent=False, active=()):
    runroot = Path(runroot).resolve()
    work = runroot / "h100_sync"
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "sync.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = build_plan(runroot, names, work)
        plan["arms_sha256"] = sha(runroot / "arms.json")
        plan["worker_host"] = worker_host()
        if plan_only:
            return plan
        if any(x["size"] > 8 * (1 << 30) for x in plan["files"]):
            raise ValueError("single asset exceeds bounded RPC budget; split it before sync")
        protected = {}
        for chain in active:
            active_plan = json.loads((chain["root"] / "h100_sync/synced.json").read_text())
            for item in active_plan["files"]:
                if item["rel"] in protected and protected[item["rel"]] != item["sha256"]:
                    raise RuntimeError("ACTIVE_ASSET_CONFLICT " + item["rel"])
                protected[item["rel"]] = item["sha256"]
        for item in plan["files"]:
            if item["rel"] in protected and item["sha256"] != protected[item["rel"]]:
                raise RuntimeError("ACTIVE_ASSET_CONFLICT " + item["rel"])
        with daemon(work, plan["files"]) as url:
            spec = dict(dest=str(BASE), files=plan["files"], reserve_files=plan["files"], url=url)
            if concurrent:
                spec["protected"] = protected
            pull_action = "pull-new" if concurrent else "pull"
            # One RPC must stay below tether's ten-minute cap. Chunk by 8 GiB.
            batch, size = [], 0
            for item in plan["files"]:
                if batch and size + item["size"] > 8 * (1 << 30):
                    print(rpc("h100", pull_action, {**spec, "files": batch}, timeout=590))
                    batch, size = [], 0
                if item["size"] > 8 * (1 << 30):
                    raise ValueError(f"single asset exceeds bounded RPC budget: {item['rel']}")
                batch.append(item)
                size += item["size"]
            if batch:
                print(rpc("h100", pull_action, {**spec, "files": batch}, timeout=590))
        # Matrices remain os_cl/cfg relative, preserving run_gtp's contract.
        with tempfile.TemporaryDirectory(prefix="oscl_sb_arms_") as td:
            pairs = []
            for row in plan["arms"]:
                original = next(r for r in json.loads((runroot / "arms.json").read_text()) if r["arm"] == row["arm"])
                cfg = next(x for x in plan["files"] if x["original"] == original["yaml"])
                pairs += [(Path(cfg["source"]), Path("cfg") / (row["arm"] + ".yaml")),
                          (Path(original["matrix"]), Path("cfg") / ("matrix_" + row["arm"] + ".yaml"))]
            bundle(worker_host(), island() / "os_cl", pairs)
        (work / "synced.json").write_text(json.dumps(plan, indent=1))
        print("SYNC_OK")
        return plan


def collect(runroot, arm, manifest=None):
    root = Path(runroot).resolve()
    target = root / "runs" / arm
    tag = root.name
    host = launch_host(root, arm)
    for node, src, prefix in ((host, island(host) / "os_cl/runs" / tag / arm, "client"),
                              ("h100", BASE / "runs" / tag / "runs" / arm, "")):
        archive = (BASE / ".rpc" if node == "h100" else T108_STAGE if node == "timan108" else T107_STAGE) / f"collect_{tag}_{arm}.tar"
        meta = json_output(rpc(node, "archive", dict(root=str(src), archive=str(archive), prefix=prefix)))
        with tempfile.TemporaryDirectory(prefix="oscl_sb_collect_") as td:
            local = Path(td) / "logs.tar"
            for attempt in range(8):
                try:
                    run(["tether", "pull", "--force", f"{node}:{archive}", local], timeout=590, attempts=1)
                    break
                except (RuntimeError, subprocess.TimeoutExpired):
                    if attempt == 7:
                        raise
                    time.sleep(min(15 * (attempt+1), 60))
            if sha(local) != meta["sha256"]:
                raise RuntimeError("collection archive SHA mismatch")
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(local) as tf:
                for m in tf.getmembers():
                    p = Path(m.name)
                    if not m.isfile() or p.is_absolute() or ".." in p.parts:
                        raise ValueError("unsafe collected archive member")
                    dest = target / p
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(m) as f:
                        dest.write_bytes(f.read())
        remote(node, ["rm", "-f", archive])
    from exp.offline_search.closed_loop.ops.collect import summarize
    result = summarize(root, arm, manifest=manifest)
    print(f"COLLECT_OK arm={arm} complete={result['complete']} success={result['success']} sr={result['sr']}")
    return result


def launch_host(root, arm):
    path = Path(root) / "runs" / arm / "h100_launch.json"
    if path.exists():
        host = json.loads(path.read_text()).get("worker_host", "timan108")
    else:
        host = worker_host()
    if host not in FLEETS:
        raise ValueError("unknown worker host in launch receipt")
    return host


def ports_workers():
    cap = FLEETS[worker_host()][2]
    ports = [int(p) for p in os.environ["PORTS"].split(",")]
    wps = int(os.environ.get("WPS", "8"))
    if len(set(ports)) != len(ports) or any(not 23200 <= p <= 23299 for p in ports):
        raise ValueError("PORTS must be distinct ports in 23200..23299")
    if wps < 1 or wps * len(ports) > cap:
        raise ValueError(f"WPS * number of ports must be 1..{cap}")
    return ports, wps


def selection(root, row):
    from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest, check_manifest
    manifest = os.environ.get("OSCL_MANIFEST") or row.get("manifest")
    if manifest:
        m = load_manifest(str(manifest).replace("<RUN>", str(root)))
        check_manifest(m, row["model"], row["suite"])
        return len(m["selected"]), m["path"], m["sha256"]
    def ids(name, n):
        val = os.environ.get(name, "")
        selected = [int(v) for v in val.split(",")] if val else list(range(n))
        if len(set(selected)) != len(selected) or any(v < 0 or v >= n for v in selected):
            raise ValueError(f"invalid {name}")
        return selected
    return len(ids("OSCL_EPISODES", 50)) * len(ids("OSCL_TASKS", 10)), None, None


def server_spec(root, row, port):
    full = row.get("full_model", False) or os.environ.get("STAGE1_ONLY", "1") == "0"
    need = int(os.environ.get("NEED_MB", "8000" if row["model"] == "groot" and full else "9000" if full else "3000"))
    if "NEED_MB" not in os.environ and row.get("method", "").endswith("clip.method:ClipAWM"):
        need += 1000  # Separate float32 image tower; leave room for its lazy first-query load.
    env = {k: os.environ[k] for k in ("OMP", "GPU_LOCK", "GROOT_DENOISING_STEPS", "PI05_CKPT", "GROOT_CKPT") if k in os.environ}
    env.update({k: str(v) for k, v in row.get("server_env", {}).items()})
    if any(k.startswith(("OSDEBUG", "OSCL_DEBUG")) for k in env):
        raise ValueError("debug environment forbidden in standard mode")
    env.update(STAGE1_ONLY="0" if full else "1", STOCK="1" if row["mode"] == "stock" else "0")
    args = []
    if row["mode"] != "stock":
        args = ["--os-method", row["method"] if row["mode"] == "plugin" else "native",
                "--os-cell", row["cell"], *row.get("plugin_args", [])]
        if row["mode"] == "plugin":
            args += ["--os-kwargs", json.dumps(row.get("kwargs") or {})]
        if row.get("server_seed") is not None:
            args += ["--os-seed", str(row["server_seed"] * 65536 + port)]
    return dict(role="server", out=str(BASE / "runs" / root.name / "runs" / row["arm"] / f"server_{port}"),
                tag=f"{row['arm']}_{port}", model=row["model"], suite=row["suite"], port=port,
                yaml=row["yaml"], plugin=args, env=env, need_mb=need)


def chain(runroot, names):
    root = Path(runroot).resolve()
    host, worker_island = worker_host(), island()
    ports, wps = ports_workers()
    attempts = int(os.environ.get("MAX_ATTEMPTS", "3"))
    if attempts < 1:
        raise ValueError("MAX_ATTEMPTS must be positive")
    poll_seconds = int(os.environ.get("POLL_SECONDS", str(POLL_SECONDS)))
    failure_limit = int(os.environ.get("STATUS_FAILURE_LIMIT", str(STATUS_FAILURE_LIMIT)))
    if poll_seconds < 1 or failure_limit < 1:
        raise ValueError("POLL_SECONDS and STATUS_FAILURE_LIMIT must be positive")
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    log = root / "runs/chain.log"
    def event(msg):
        line = "EV " + time.strftime("%m-%d_%H:%M:%S") + " " + msg
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")
    def note(msg):
        with open(log, "a") as f:
            f.write("   " + time.strftime("%H:%M:%S") + " " + msg + "\n")
        print(msg, flush=True)
    active = dict(servers=[], driver=None)
    def cleanup():
        errors = []
        if active["driver"]:
            try:
                note(rpc(host, "stop", active["driver"], timeout=590))
                active["driver"] = None
            except Exception as exc:
                errors.append(str(exc))
        for spec in active["servers"][:]:
            try:
                note(rpc("h100", "stop", spec))
                active["servers"].remove(spec)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("cleanup failed: " + "\n".join(errors))
    def interrupted(sig, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    arm = "preflight"
    with fleet_lock(root):
        # Read the plan under the same lock as sync, avoiding a stale preflight
        # snapshot if maintenance finishes just before this controller acquires it.
        plan = json.loads((root / "h100_sync/synced.json").read_text())
        if plan.get("worker_host", "timan108") != host:
            raise RuntimeError("synced worker host differs; sync for WORKER_HOST again")
        if plan["arms_sha256"] != sha(root / "arms.json"):
            raise RuntimeError("arms.json changed since sync; sync again")
        prepared = {r["arm"]: r for r in plan["arms"]}
        originals = {r["arm"]: r for r in json.loads((root / "arms.json").read_text())}
        missing = sorted(set(names) - prepared.keys())
        if missing:
            raise RuntimeError("requested arms missing from synced.json: " + ", ".join(missing))
        missing = sorted(set(names) - originals.keys())
        if missing:
            raise RuntimeError("requested arms missing from arms.json: " + ", ".join(missing))
        receipt = dict(pid=os.getpid(), starttime=identity(os.getpid()),
                       cmdline=Path(f"/proc/{os.getpid()}/cmdline").read_bytes().hex(), worker_host=host)
        (state / "h100_chain.owner.json").write_text(json.dumps(receipt))
        try:
            code_tree = json_output(rpc("h100", "manifest", dict(root=str(BASE / "openpi")), timeout=590))
            if len(code_tree["sha256"]) != 64 or code_tree["files"] < 1:
                raise RuntimeError("invalid/empty h100 code-tree manifest")
            pending = []
            def read_status(pairs):
                statuses, failed = [], False
                for machine, spec in pairs:
                    try:
                        # The next poll supplies the retry/backoff for status reads; avoid
                        # multiplying five per-poll deadlines during a node outage.
                        status = json_output(rpc(machine, "status", spec, timeout=60, attempts=1))
                        if not isinstance(status, dict) or any(
                                not isinstance(status.get(key), bool) for key in ("running", "dead", "listening")):
                            raise ValueError("invalid status reply")
                        statuses.append(status)
                    except Exception as exc:
                        statuses.append(None)
                        failed = True
                        note(f"STATUS_UNKNOWN node={machine} role={spec['role']} error={str(exc)[:300]}")
                return statuses, failed
            def status_failures(failures, failed):
                failures = failures + 1 if failed else 0
                if failures >= failure_limit:
                    raise RuntimeError(f"STATUS_POLL_FAILED consecutive={failures}")
                return failures
            for arm in names:
                row = prepared[arm]
                expect, manifest, manifest_sha = selection(root, originals[arm])
                done = state / (f"{arm}.manifest_{manifest_sha}.DONE" if manifest else f"{arm}.DONE")
                if done.exists():
                    note(f"skip {arm} (DONE)")
                    continue
                (state / "CHAIN.DONE").unlink(missing_ok=True)
                dest = root / "runs" / arm
                dest.mkdir(parents=True, exist_ok=True)
                env = {k: os.environ[k] for k in ("OSCL_EPISODES", "OSCL_TASKS") if k in os.environ}
                if manifest:
                    saved = dest / "manifest.json"
                    saved.write_bytes(Path(manifest).read_bytes())
                    remote_manifest = str(worker_island / "os_cl/cfg" / f"manifest_{root.name}_{arm}_{manifest_sha}.json")
                    push(host, saved, remote_manifest)
                    env["OSCL_MANIFEST"] = remote_manifest
                    manifest = str(saved)
                (dest / "selection.json").write_text(json.dumps(dict(manifest=manifest)))
                if row.get("replan_steps"):
                    env["OSCL_REPLAN_STEPS"] = str(row["replan_steps"])
                driver = dict(role="driver", out=str(worker_island / "os_cl/runs" / root.name / arm), arm=arm,
                              suite=row["suite"], servers=[f"149.165.153.233:{p}" for p in ports],
                              workers=[wps] * len(ports), env=env)
                if host != "timan108":
                    env["WORKER_HOST"] = host
                specs = [server_spec(root, row, p) for p in ports]
                def save_launch():
                    (dest / "h100_launch.json").write_text(json.dumps(
                        dict(servers=specs, driver=driver, code_tree=code_tree, worker_host=host), indent=1))
                save_launch()
                (state / "current").write_text(arm + "\n")
                event(f"ARM_START arm={arm} suite={row['suite']} ports={','.join(map(str,ports))} workers={wps*len(ports)} expect={expect}")
                def servers_up():
                    free = int(remote("h100", ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).splitlines()[-1])
                    if free < sum(x["need_mb"] for x in specs):
                        raise RuntimeError(f"GPU_TIGHT free={free} need={sum(x['need_mb'] for x in specs)}")
                    for spec in specs:
                        spec["launch_id"] = uuid.uuid4().hex
                        # Record before launching so partially failed starts are still cleaned up.
                        active["servers"].append(spec)
                        save_launch()
                        note(rpc("h100", "start", spec))
                    save_launch()
                    deadline = time.monotonic() + 1200
                    failures = 0
                    while True:
                        statuses, failed = read_status([("h100", x) for x in specs])
                        failures = status_failures(failures, failed)
                        if any(s and s["dead"] for s in statuses):
                            raise RuntimeError("SERVER_DIED_AT_BOOT " + json.dumps(statuses))
                        if all(s and s["listening"] and s["running"] for s in statuses):
                            break
                        if time.monotonic() > deadline:
                            raise RuntimeError("SERVER_BOOT_TIMEOUT")
                        time.sleep(5)
                    event(f"SERVERS_READY arm={arm}")
                servers_up()
                complete = False
                for attempt in range(1, attempts+1):
                    driver["launch_id"] = uuid.uuid4().hex
                    save_launch()
                    active["driver"] = driver
                    note(rpc(host, "start", driver))
                    note(f"driver attempt {attempt} launched")
                    dead = False
                    failures = 0
                    while True:
                        time.sleep(poll_seconds)
                        statuses, failed = read_status([*(("h100", x) for x in specs), (host, driver)])
                        failures = status_failures(failures, failed)
                        server_statuses, driver_status = statuses[:-1], statuses[-1]
                        dead = any(s and (s["dead"] or not s["running"]) for s in server_statuses)
                        if dead:
                            event(f"SERVER_DIED arm={arm}")
                            note(rpc(host, "stop", driver, timeout=590))
                            active["driver"] = None
                            break
                        if driver_status and not driver_status["running"]:
                            # Session disappearance alone does not prove legacy run_gtp has exited.
                            note(rpc(host, "stop", driver, timeout=590))
                            active["driver"] = None
                            break
                    purged = remote(host, ["python3", worker_island / "os_cl/purge_exc.py", driver["out"]])
                    note("purged=" + purged)
                    if int(purged.splitlines()[-1]):
                        event(f"EXC_PURGED arm={arm} n={purged.splitlines()[-1]}")
                    cmd = ["python3", worker_island / "os_cl/count.py", Path(driver["out"]) / "journal.jsonl", "--arm", arm]
                    if env.get("OSCL_MANIFEST"):
                        cmd += ["--manifest", env["OSCL_MANIFEST"]]
                    n, s, rows = map(int, remote(host, cmd).splitlines()[-1].split())
                    note(f"journal: complete={n} success={s} rows={rows}")
                    if n >= expect:
                        cleanup()
                        try:
                            result = collect(root, arm, manifest)
                            if result["complete"] < expect:
                                raise RuntimeError("COLLECT_FAILED completion count changed")
                        except Exception as exc:
                            pending.append(arm)
                            (state / f"{arm}.ERROR").write_text(f"COLLECT_PENDING: {exc}\n")
                            event(f"COLLECT_PENDING arm={arm} error={type(exc).__name__}:{str(exc)[:1000]}")
                        else:
                            done.touch()
                            (state / f"{arm}.ERROR").unlink(missing_ok=True)
                            event(f"ARM_DONE arm={arm} complete={n} success={s} sr={s/max(n,1):.3f}")
                        complete = True
                        break
                    event(f"ARM_INCOMPLETE arm={arm} attempt={attempt} complete={n} -> resume")
                    if dead and attempt < attempts:
                        cleanup()
                        servers_up()
                        event(f"SERVERS_RESTARTED arm={arm}")
                if not complete:
                    raise RuntimeError(f"ARM_FAILED arm={arm} after {attempts} attempts")
            (state / "current").write_text("none\n")
            if pending:
                (state / "CHAIN.ERROR").write_text("COLLECT_PENDING: " + " ".join(pending) + "\n")
                (state / "CHAIN.DONE").unlink(missing_ok=True)
            else:
                (state / "CHAIN.ERROR").unlink(missing_ok=True)
                (state / "CHAIN.DONE").touch()
            event("CHAIN_DONE " + " ".join(names) + (" collect_pending=" + ",".join(pending) if pending else ""))
        except BaseException as exc:
            event(f"CHAIN_STOPPED at {arm} error={type(exc).__name__}:{str(exc)[:1000]}")
            (state / f"{arm}.ERROR").write_text(str(exc) + "\n")
            (state / "CHAIN.ERROR").write_text(f"stopped at {arm}: {exc}\n")
            try:
                cleanup()
            finally:
                (state / "current").write_text("none\n")
            raise


def abort(root, arms):
    receipt_path = Path(root) / "state/h100_chain.owner.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if identity(receipt["pid"]) == receipt["starttime"]:
            cmdline = Path(f"/proc/{receipt['pid']}/cmdline").read_bytes().hex()
            if cmdline != receipt["cmdline"]:
                raise RuntimeError("local chain PID has a foreign command; refusing")
            os.kill(receipt["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 550
            while identity(receipt["pid"]) == receipt["starttime"] and time.monotonic() < deadline:
                time.sleep(1)
            if identity(receipt["pid"]) == receipt["starttime"]:
                raise RuntimeError("local chain did not stop; refusing to race its relaunch loop")
    for arm in arms:
        spec = json.loads((Path(root) / "runs" / arm / "h100_launch.json").read_text())
        print(rpc(launch_host(Path(root), arm), "stop", spec["driver"], timeout=590))
        for server in spec["servers"]:
            print(rpc("h100", "stop", server))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["setup", "plan", "sync", "chain", "collect", "abort", "verify", "verify-worker", "render"])
    ap.add_argument("--worker-only", action="store_true", help="setup only the selected worker island")
    ap.add_argument("--concurrent", action="store_true", help="sync without replacing any existing h100 asset")
    ap.add_argument("run_root", nargs="?")
    ap.add_argument("arms", nargs="*")
    a = ap.parse_intermixed_args()
    if a.action == "setup":
        setup(worker_only=a.worker_only)
    elif a.action in ("plan", "sync"):
        sync(a.run_root, a.arms, a.action == "plan", concurrent=a.concurrent)
    elif a.action == "chain":
        chain(a.run_root, a.arms)
    elif a.action == "collect":
        for arm in a.arms:
            collect(a.run_root, arm)
    elif a.action == "abort":
        abort(a.run_root, a.arms)
    elif a.action == "verify":
        expected = json.loads(Path(a.run_root or "/home/weiland/trace_runs/os_closed_loop/r08_fits/checkpoint_sha.json").read_text())
        print(rpc("h100", "checkpoints", expected, timeout=590))
        for model, py in (("pi05", "/home/exouser/openpi/.venv/bin/python"),
                          ("groot", "/home/exouser/gr00t_n15_venv/.venv/bin/python")):
            env = ["env", "HOME=/home/exouser", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                   f"PYTHONPATH=/home/exouser/gr00t_n15:/home/exouser/gr00t_n15/examples/Libero:{BASE}/openpi:{BASE}/openpi/src:{BASE}/openpi/packages/openpi-client/src"]
            print(remote("h100", [*env, py, BASE / "openpi/exp/offline_search/closed_loop/ops/h100/probe.py", model], timeout=300))
    elif a.action == "render":
        print(remote(worker_host(), ["env", "HOME=/home/zixuans8", "LIBERO_CONFIG_PATH=" + str(island() / "os_cl/libero"),
                                  "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                                  "python3", island() / "os_cl/render_test.py", "--gpus", ",".join(map(str, FLEETS[worker_host()][1]))], timeout=590))
    elif a.action == "verify-worker":
        print(remote(worker_host(), ["python3", island() / "os_cl/verify_worker.py"], timeout=180))


if __name__ == "__main__":
    main()
