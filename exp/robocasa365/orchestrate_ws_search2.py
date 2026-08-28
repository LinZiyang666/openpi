"""Thin ws2 orchestrator: persistent dynamic-bundle servers + one driver.

Round 1 needed a slot scheduler because every cell required a server restart;
ws2 servers stay up for a whole phase (bundle hot-swap over the wire), so
orchestration collapses to a handful of idempotent operations, each a
subcommand:

- ``servers-up``    serially start N dynamic-bundle server tmux sessions on the
                    serving host — the entry point, interpreter and checkpoint
                    come from ``--teacher`` (``TEACHERS``) — gating each on
                    (a) free RAM, (b) free VRAM against that teacher's measured
                    per-server need, (c) a fresh log showing THIS bootstrap
                    yaml path, and (d) the real
                    ``INFO:websockets.server:server listening`` line — the CLI
                    banner prints before bind and is not trusted (round-1
                    lesson). Both teachers emit that line: they share
                    ``WebsocketPolicyServer``, which raises the
                    ``websockets.server`` logger to INFO.
- ``servers-down``  kill only OUR tmux sessions and any serve process whose
                    /proc cmdline carries the teacher's entry point + one of
                    our ports (PID-anchored; no broad pkill on a shared host).
- ``driver-up``     launch ``run_ws_search2 --role driver`` in tmux on the
                    serving host.
- ``agents-up``     launch worker agents (``--role agent``) on a worker host,
                    one tmux per fleet.
- ``agents-down``   stop ONE fleet: kill its agent tmux FIRST (the agent is a
                    supervisor and would respawn workers), then sweep surviving
                    ``worker_entry`` PIDs whose cmdline matches ALL of the
                    fleet's identity — entry point, ``--server-key``,
                    ``--driver-host`` and ``--driver-port`` — then report
                    leftovers. Sibling fleets share the driver endpoint, so the
                    server key is what keeps this fleet-scoped.
- ``status``        one-line health: tmux presence, port listeners, journal
                    line count, summary complete count.

Everything host-specific rides CLI flags with the planned-deployment defaults
(the throwaway serving clone; ports 23160+). All remote execution goes through
``tether exec <node> -- bash -lc '...'``; commands are printed before running
so a dry ``--echo`` run doubles as a runbook.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time

SERVE_DEFAULTS = {
    "node": "weilandserver",
    "repo": "/data/openpi_text_ivf_build",
}

# Per-teacher serving recipe. The two teachers do not share an entry point, an
# interpreter or a flag spelling, so every teacher-specific string lives here
# and nowhere else; ``--teacher`` picks a row and the subcommands read it.
#
# ``vram_mb`` is the per-server VRAM need INCLUDING headroom, carried over from
# round 1's measured footprints (GR00T ~6.25 GiB, pi0.5 ~8 GiB). It is the
# floor the three-read admission gate enforces, so a pool that no longer fits
# the card is refused at the next server rather than OOM-ing the running ones.
TEACHERS: dict[str, dict[str, str | int]] = {
    "groot_tp": {
        "entry": "exp/robocasa365/serve_groot_n15.py",
        # /proc-cmdline anchor for the shutdown sweep: must be the token that
        # identifies THIS teacher's serve process and no other.
        "anchor": "serve_groot_n15",
        "python": "/home/weiland/gr00t_n15_venv/.venv/bin/python",
        # {repo} is filled from --repo so a different serving clone stays consistent.
        "pythonpath": "/home/weiland/gr00t_n15:{repo}/src:{repo}",
        "checkpoint": (
            "/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/"
            "target_posttraining/atomic_seen/checkpoint-60000"
        ),
        "vram_mb": 8500,
    },
    "pi05": {
        "entry": "scripts/serve_policy.py",
        "anchor": "serve_policy.py",
        # The throwaway serving clone has NO venv of its own (verified on the
        # host), so pi0.5 borrows the main checkout's interpreter and grafts
        # the clone's source in front of it via PYTHONPATH -- the same trick
        # GR00T uses. Without the graft the server would import the main
        # checkout's openpi and reject this round's pooling knobs.
        "python": "/home/weiland/openpi/.venv/bin/python",
        "pythonpath": "{repo}/src:{repo}",
        "checkpoint": "/home/weiland/ckpt_pi05_robocasa_pytorch",
        "vram_mb": 10500,
    },
}


def teacher_spec(teacher: str, repo: str) -> dict[str, str | int]:
    """The serving row for ``teacher``, with ``{repo}`` resolved."""
    try:
        spec = TEACHERS[teacher]
    except KeyError:
        raise SystemExit(
            f"no serving recipe for teacher {teacher!r}; known: {sorted(TEACHERS)}"
        ) from None
    return {k: (v.format(repo=repo) if isinstance(v, str) else v) for k, v in spec.items()}


def serve_command(args: argparse.Namespace, port: int, log: str) -> str:
    """The remote shell line that starts ONE server for ``args.teacher``.

    The two recipes differ in more than a path. GR00T's server takes
    ``--cache-config`` and an explicit ``--allow-dynamic-bundles``; pi0.5 goes
    through ``scripts/serve_policy.py``, whose flag is ``--cache_config`` and
    which must receive every flag BEFORE the ``policy:checkpoint`` subcommand
    (tyro ordering) — a flag after it is silently parsed as the subcommand's.
    pi0.5 needs no ``--allow-dynamic-bundles``: ``WebsocketPolicyServer``
    defaults it to True and ``serve_policy`` never overrides it (verified in
    plan §1), which is what lets the pool hot-swap bundles without restarts.
    """
    spec = teacher_spec(args.teacher, args.repo)
    prefix = (
        f"cd {args.repo} && CUDA_VISIBLE_DEVICES={args.cuda_device} OPENPI_MONITOR_LEVEL=BASIC "
    )
    if args.pythonpath:
        prefix += f"PYTHONPATH={args.pythonpath} "
    if args.teacher == "groot_tp":
        body = (
            f"{args.python} {spec['entry']} --checkpoint {args.checkpoint} --port {port} "
            f"--concurrent --allow-dynamic-bundles --cache-config {args.bootstrap_yaml}"
        )
    else:
        body = (
            f"{args.python} {spec['entry']} --port {port} "
            f"--cache_config {args.bootstrap_yaml} policy:checkpoint "
            f"--policy.config pi05_robocasa --policy.dir {args.checkpoint}"
        )
    return f"{prefix}{body} 2>&1 | tee {log}"


def run(cmd: list[str], *, echo_only: bool) -> str:
    """Run one local command, echoing it first so a dry run reads as a runbook."""
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if echo_only:
        return ""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    out = (proc.stdout + proc.stderr).strip()
    if out:
        print(out, flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed (rc={proc.returncode})")
    return out


def texec(node: str, script: str, *, echo_only: bool, home: str = "/home/weiland") -> str:
    """Run a script on a tether node under an explicit HOME.

    The tether agent's default HOME is its own state directory, not the user's,
    so every remote script has to set it — and the right value is per NODE:
    the serving host runs as ``weiland`` while the worker island runs as a
    different account entirely, whose venv, tmux socket and caches all live
    under its own home.
    """
    return run(["tether", "exec", node, "--", "bash", "-lc",
                f"export HOME={home}; {script}"], echo_only=echo_only)


# ERE metacharacters that must not act as pattern syntax when an identity
# value (a hostname, an endpoint) is spliced into a sweep pattern.
_ERE_SPECIAL = set(r".[]\*+?{}()|^$")


def ere_literal(value: str) -> str:
    """Escape ``value`` so it matches itself in an extended regex.

    Identity values carry dots: an unescaped ``ziyanglin.com`` also matches
    ``ziyanglinXcom``, and a sweep is a kill list — a pattern that matches more
    than the literal it came from is how another session's fleet dies.
    """
    return "".join("\\" + ch if ch in _ERE_SPECIAL else ch for ch in value)


def preflight_gate(args: argparse.Namespace, port: int, session: str) -> None:
    """Shared-host admission checks, run before every server we add.

    Four gates, all from the plan's risk register: the port and tmux name must
    be free (23100-23199 and the session namespace are shared with other
    sessions on this box), free RAM must clear the floor (each pooled server
    keeps ~21 GB resident), free VRAM must clear the floor on THREE consecutive
    reads (a single sample catches another process mid-allocation), and the GPU
    must already be warm (this card computes silently wrong from cold).
    """
    gpu = getattr(args, "cuda_device", 0)
    # -i <gpu>: with more than one card the pool is spread across them, so the
    # admission read has to be of the card this server will actually use.
    probe = texec(
        args.node,
        f"ss -tln | grep -c ':{port} ' || true; "
        f"tmux has-session -t '={session}' 2>/dev/null && echo TAKEN || echo FREE; "
        "free -g | awk '/Mem:/{print $7}'; "
        f"nvidia-smi -i {gpu} --query-gpu=memory.free,temperature.gpu --format=csv,noheader,nounits; "
        f"sleep 2; nvidia-smi -i {gpu} --query-gpu=memory.free --format=csv,noheader,nounits; "
        f"sleep 2; nvidia-smi -i {gpu} --query-gpu=memory.free --format=csv,noheader,nounits",
        echo_only=False,
    ).splitlines()
    rows = [line.strip() for line in probe if line.strip()]
    if len(rows) < 6:
        raise SystemExit(f"preflight probe returned {len(rows)} lines, expected 6: {rows}")
    port_used, session_state, free_ram, first_gpu, vram2, vram3 = rows[-6:]
    if port_used != "0":
        raise SystemExit(f"port {port} is already listening on {args.node}; pick another block")
    if session_state != "FREE":
        raise SystemExit(f"tmux session {session!r} already exists on {args.node}; pick another prefix")
    if int(free_ram) < args.min_free_ram_gb:
        raise SystemExit(f"free RAM {free_ram}G < {args.min_free_ram_gb}G floor; not adding :{port}")
    vram1, temp = (v.strip() for v in first_gpu.split(","))
    reads = [int(vram1), int(vram2), int(vram3)]
    if min(reads) < args.min_free_vram_mb:
        raise SystemExit(
            f"free VRAM three-read min {min(reads)}MiB < {args.min_free_vram_mb}MiB floor "
            f"(reads {reads}); not adding :{port}"
        )
    if args.min_gpu_temp_c and int(temp) < args.min_gpu_temp_c:
        # Only meaningful on hardware with the cold-miscompute defect; pass
        # --min-gpu-temp-c 0 on a healthy card rather than warming it for show.
        raise SystemExit(
            f"GPU {gpu} at {temp}C is below the {args.min_gpu_temp_c}C warm floor — a card "
            "with the cold-miscompute defect must be warmed before it serves"
        )
    print(f"[orch] :{port} preflight ok (gpu {gpu}, RAM {free_ram}G, VRAM {reads}MiB, {temp}C)",
          flush=True)


def resolve_serve_defaults(args: argparse.Namespace) -> None:
    """Fill the unset serving flags from the teacher's row, in place.

    The interpreter, PYTHONPATH, checkpoint and VRAM floor are all functions of
    the teacher, but argparse resolves subparser defaults before ``--teacher``
    is known. So they default to None on the CLI and are bound here, which also
    keeps an explicit override winning over the table.
    """
    spec = teacher_spec(args.teacher, args.repo)
    for flag, key in (("python", "python"), ("pythonpath", "pythonpath"),
                      ("checkpoint", "checkpoint"), ("min_free_vram_mb", "vram_mb")):
        if getattr(args, flag, None) is None:
            setattr(args, flag, spec[key])


def cmd_servers_up(args: argparse.Namespace) -> None:
    """Start the server pool serially, each gated on RAM and a real bind line."""
    resolve_serve_defaults(args)
    ports = [int(p) for p in args.ports.split(",")]
    for port in ports:
        session = f"{args.tmux_prefix}{port}"
        log = f"/tmp/{session}.log"
        if not args.echo:
            preflight_gate(args, port, session)
        serve = serve_command(args, port, log)
        texec(args.node,
              f"tmux new -s {session} -d {shlex.quote(serve)}",
              echo_only=args.echo)
        if args.echo:
            continue
        # Readiness: a fresh log naming OUR yaml AND the real bind line. The
        # CLI banner prints before bind and is not trusted (round-1 lesson).
        # ``|| true`` is load-bearing: grep -c exits 1 on a zero count, and an
        # un-neutralised non-zero would make the FIRST not-yet-ready poll abort
        # the launch instead of polling.
        deadline = time.time() + args.ready_timeout_s
        while True:
            probe = texec(
                args.node,
                f"(grep -c -- {shlex.quote(args.bootstrap_yaml)} {log} 2>/dev/null || true); "
                f"(grep -c 'INFO:websockets.server:server listening' {log} 2>/dev/null || true)",
                echo_only=False,
            ).splitlines()
            counts = [line.strip() for line in probe if line.strip().isdigit()]
            if len(counts) >= 2 and counts[-2] != "0" and counts[-1] != "0":
                print(f"[orch] :{port} ready", flush=True)
                break
            if time.time() > deadline:
                raise SystemExit(f":{port} not ready in {args.ready_timeout_s}s — inspect {log}")
            time.sleep(10)


def cmd_servers_down(args: argparse.Namespace) -> None:
    """Kill our tmux sessions, then sweep leftovers anchored on entry point + port."""
    ports = [int(p) for p in args.ports.split(",")]
    for port in ports:
        session = f"{args.tmux_prefix}{port}"
        texec(args.node, f"tmux kill-session -t '={session}' 2>/dev/null || true", echo_only=args.echo)
    # PID-anchored sweep on the ACTUAL cmdline shape: the serve process carries
    # "--port <N>" (no colon) and the repo appears only via PYTHONPATH/cwd, not
    # necessarily as an argv token -- an earlier pattern required both a colon
    # and a literal repo path and therefore swept nothing, leaving ~21G
    # orphans that only surfaced when the next servers-up hit its RAM gate.
    # Two independent anchors are still required: the entry-point name AND one
    # of OUR ports. Never a broad pkill on a shared machine.
    # ``[-]-port`` not ``--port``: a pattern starting with a dash is read as a
    # grep OPTION, so the bracketed first character is what makes the anchor
    # match at all (verified against a real serve cmdline).
    port_anchor = "|".join(f"[-]-port {p}( |$)" for p in ports)
    anchor = teacher_spec(args.teacher, args.repo)["anchor"]
    sweep = (
        "for pid in $(ls /proc | grep -E '^[0-9]+$'); do "
        "c=$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null); "
        f"case \"$c\" in *{anchor}*) "
        f"printf '%s' \"$c\" | grep -Eq '{port_anchor}' && kill $pid && "
        "echo \"swept $pid\";; esac; done; true"
    )
    texec(args.node, sweep, echo_only=args.echo)


def cmd_driver_up(args: argparse.Namespace) -> None:
    """Launch the phase driver in tmux on the serving host."""
    drive = (
        f"cd {args.repo} && PYTHONPATH=src:. {args.driver_python} -m exp.robocasa365.run_ws_search2 "
        f"--teacher {args.teacher} --servers {args.servers} --run-prefix {args.run_prefix} "
        f"--config-dir {args.config_dir} --env-config {args.env_config} "
        f"--episodes {args.episodes} --role driver --bind-host 0.0.0.0 --bind-port {args.driver_port} "
        + (f"--manifest {args.manifest} " if args.manifest else "")
        + f"{args.extra_args} 2>&1 | tee /tmp/{args.tmux_prefix}driver.log"
    )
    texec(args.node, f"tmux new -s {args.tmux_prefix}driver -d {shlex.quote(drive)}", echo_only=args.echo)


def cmd_agents_up(args: argparse.Namespace) -> None:
    """Launch one worker fleet bound to a single endpoint of the pool."""
    session = f"{args.tmux_prefix}agent{args.fleet}"
    agent = (
        f"cd {args.worker_repo} && PYTHONPATH=src:. {args.agent_python} -m exp.robocasa365.run_ws_search2 "
        f"--teacher {args.teacher} --servers {args.servers} --run-prefix {args.run_prefix} "
        f"--config-dir {args.config_dir} --env-config {args.env_config} "
        f"--role agent --agent-server {args.agent_server} --workers {args.workers} "
        f"--gpu-ids {args.gpu_ids} --driver-host {args.driver_host} --driver-port {args.driver_port} "
        + (f"--manifest {args.manifest} " if args.manifest else "")
        + f"2>&1 | tee /tmp/{session}.log"
    )
    texec(args.worker_node, f"tmux new -s {session} -d {shlex.quote(agent)}",
          echo_only=args.echo, home=args.worker_home)


def cmd_agents_down(args: argparse.Namespace) -> None:
    """Stop one worker fleet: the supervisor first, then its workers.

    Order matters. ``WorkerAgent`` is a supervisor: killing workers alone makes
    it respawn them, and a worker that exits on the final batch's shutdown is
    restarted the same way. So the tmux session (the agent) goes first, then
    any surviving ``worker_entry`` process is swept -- anchored on BOTH the
    entry point and this fleet's driver endpoint, because the worker fleets of
    other sessions run the same module on this shared machine. Never a broad
    pkill. Idempotent: re-running on an already-clean host is a no-op.
    """
    session = f"{args.tmux_prefix}agent{args.fleet}"
    texec(args.worker_node, f"tmux kill-session -t '={session}' 2>/dev/null || true",
          echo_only=args.echo, home=args.worker_home)
    # Fleet-level identity, not phase-level. Every fleet of this phase points at
    # the SAME driver, so the driver endpoint alone would sweep the other
    # fleets' workers off this node too; what distinguishes a fleet is the
    # server endpoint its workers are bound to (``--server-key``). All three
    # must match. ``[-]-x`` not ``--x``: a pattern starting with a dash is read
    # as a grep OPTION (the trap the server sweep already hit).
    anchors = [
        f"[-]-server-key {ere_literal(args.agent_server)}( |$)",
        f"[-]-driver-host {ere_literal(args.driver_host)}( |$)",
        f"[-]-driver-port {ere_literal(str(args.driver_port))}( |$)",
    ]
    conjunction = " | ".join(f"grep -Eq '{a}' && printf '%s' \"$c\" " for a in anchors)
    match = conjunction.rsplit("&& printf", 1)[0] + "&& true"
    sweep = (
        "for pid in $(ls /proc | grep -E '^[0-9]+$'); do "
        "c=$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null); "
        "case \"$c\" in *robocasa365.worker_entry*) "
        f"if printf '%s' \"$c\" | {match}; then kill $pid && echo \"swept $pid\"; fi;; "
        "esac; done; true"
    )
    texec(args.worker_node, sweep, echo_only=args.echo, home=args.worker_home)
    # Verify, do not assume. A worker that survived the sweep keeps a GPU
    # context and an EGL surface on a shared node, so a silent success here
    # would hand the next phase a machine that looks free and is not.
    report = texec(
        args.worker_node,
        f"tmux ls 2>/dev/null | grep -c '^{session}' || true; "
        "for pid in $(ls /proc | grep -E '^[0-9]+$'); do "
        "c=$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null); "
        "case \"$c\" in *robocasa365.worker_entry*) "
        f"if printf '%s' \"$c\" | {match}; then echo LEFTOVER; fi;; "
        "esac; done; true",
        echo_only=args.echo, home=args.worker_home,
    )
    if args.echo:
        return
    lines = [line.strip() for line in report.splitlines() if line.strip()]
    sessions_left = next((int(v) for v in lines if v.isdigit()), 0)
    workers_left = sum(1 for line in lines if line == "LEFTOVER")
    if sessions_left or workers_left:
        raise SystemExit(
            f"agents-down did not fully clear fleet {args.fleet!r} on {args.worker_node}: "
            f"{sessions_left} tmux session(s), {workers_left} worker process(es) remain. "
            "Investigate before starting the next phase — a surviving worker still "
            "holds its CUDA context."
        )
    print(f"[orch] fleet {args.fleet} on {args.worker_node} is clear", flush=True)


def cmd_status(args: argparse.Namespace) -> None:
    """One-shot health read: sessions, listeners, journal size, complete cells."""
    texec(
        args.node,
        f"tmux ls 2>/dev/null | grep -c '^{args.tmux_prefix}' ; "
        f"ss -tln | grep -cE ':(23(1[6-9][0-9]))' ; "
        f"wc -l {args.data_dir}/journal_central_{args.run_prefix}.jsonl 2>/dev/null || echo 'no journal' ; "
        f"grep -l '\"complete\": true' {args.data_dir}/summary_{args.run_prefix}-*.json 2>/dev/null | wc -l",
        echo_only=args.echo,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--echo", action="store_true", help="print the remote commands without running them")
    ap.add_argument("--node", default=SERVE_DEFAULTS["node"])
    ap.add_argument("--repo", default=SERVE_DEFAULTS["repo"])
    ap.add_argument("--tmux-prefix", default="ws2s")
    ap.add_argument("--teacher", default="groot_tp", choices=sorted(TEACHERS),
                    help="serving recipe (entry point, interpreter, checkpoint, VRAM floor); "
                    "also the --teacher passed through to driver and agent runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("servers-up")
    up.add_argument("--ports", required=True, help="comma list, started serially in order")
    # None means "take the teacher's row"; an explicit value still wins. argparse
    # cannot see --teacher this early, so the binding happens after parsing (see
    # resolve_serve_defaults).
    up.add_argument("--python", default=None)
    up.add_argument("--pythonpath", default=None)
    up.add_argument("--checkpoint", default=None)
    up.add_argument("--bootstrap-yaml", required=True, help="repo-relative cell yaml for --cache-config")
    up.add_argument("--min-free-ram-gb", type=int, default=40)
    up.add_argument("--min-free-vram-mb", type=int, default=None,
                    help="floor across three consecutive reads; defaults to the teacher's "
                    "measured per-server need plus margin (GR00T 8500, pi0.5 10500)")
    up.add_argument("--min-gpu-temp-c", type=int, default=0,
                    help="cold-card floor (0 = off). Only set this on a card with the "
                    "cold-miscompute defect, where serving cold yields silently wrong results")
    up.add_argument("--cuda-device", type=int, default=0,
                    help="which GPU this server binds to; spread the pool across cards")
    up.add_argument("--ready-timeout-s", type=float, default=900.0)
    up.set_defaults(func=cmd_servers_up)

    down = sub.add_parser("servers-down")
    down.add_argument("--ports", required=True)
    down.set_defaults(func=cmd_servers_down)

    drv = sub.add_parser("driver-up")
    drv.add_argument("--servers", required=True)
    drv.add_argument("--run-prefix", default="ws2")
    drv.add_argument("--config-dir", required=True)
    drv.add_argument("--env-config", required=True)
    drv.add_argument("--episodes", type=int, default=8)
    drv.add_argument("--driver-port", type=int, default=23180)
    drv.add_argument("--driver-python", default="python3")
    drv.add_argument("--manifest", default="")
    drv.add_argument("--extra-args", default="")
    drv.set_defaults(func=cmd_driver_up)

    ag = sub.add_parser("agents-up")
    ag.add_argument("--worker-node", required=True)
    ag.add_argument("--worker-home", default="/home/weiland",
                    help="HOME on the worker node — its account is usually NOT the serving one")
    ag.add_argument("--worker-repo", required=True)
    ag.add_argument("--agent-python", default="python3")
    ag.add_argument("--servers", required=True)
    ag.add_argument("--agent-server", required=True)
    ag.add_argument("--fleet", default="0")
    ag.add_argument("--workers", type=int, default=8)
    ag.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    ag.add_argument("--run-prefix", default="ws2")
    ag.add_argument("--config-dir", required=True)
    ag.add_argument("--env-config", required=True)
    ag.add_argument("--driver-host", required=True)
    ag.add_argument("--driver-port", type=int, default=23180)
    ag.add_argument("--manifest", default="")
    ag.set_defaults(func=cmd_agents_up)

    ad = sub.add_parser("agents-down")
    ad.add_argument("--worker-node", required=True)
    ad.add_argument("--worker-home", default="/home/weiland",
                    help="HOME on the worker node — must match the one agents-up used")
    ad.add_argument("--fleet", default="0")
    ad.add_argument("--driver-host", required=True,
                    help="fleet identity part 2: the driver address its workers hold")
    ad.add_argument("--driver-port", type=int, default=23180,
                    help="fleet identity part 3: the driver port its workers hold")
    ad.add_argument("--agent-server", required=True,
                    help="fleet identity part 1: the server endpoint THIS fleet is bound "
                    "to; sibling fleets share the driver and differ only here")
    ad.set_defaults(func=cmd_agents_down)

    st = sub.add_parser("status")
    st.add_argument("--data-dir", required=True)
    st.add_argument("--run-prefix", default="ws2")
    st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
