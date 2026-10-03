"""Explicit per-arm dependency plan; no query/token corpus or whole-store copy.

Prefitted plugins carry their online arrays in their pickle. The remote runtime
additionally opens current metadata/actions and each stored payload table. Require
prefits so an unfamiliar method cannot silently demand the 365 GB fitting corpus.
"""
import copy
import json
import os
from pathlib import Path
import pickle
import re

from .node import sha

STORE = Path("/home/weiland/trace_runs/offline_search_store")
BASE = Path(os.environ.get("OSCL_H100_BASE", "/data/oscl_h100"))
ISLAND = Path(os.environ.get("OSCL_ISLAND", "/scratch/zixuans8/openpi_trace"))


def remap(value, run, store=STORE, base=BASE):
    if isinstance(value, str):
        value = value.replace("<RUN>", str(run))
        for src, dst in ((str(store), str(base / "store")), ("/dev/shm/offline_search_store", str(base / "store")),
                         (str(run), str(base / "runs" / run.name))):
            value = re.sub(re.escape(src) + r"(?=/|\Z)", lambda _: dst, value)
        # Includes external fits, calibration directories, and native library PKLs.
        if value.startswith(("/home/weiland/", "/data/")) and not value.startswith(str(base) + "/"):
            value = str(base / "mirror" / value.lstrip("/"))
        return value
    if isinstance(value, dict):
        return {k: remap(v, run, store, base) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(remap(v, run, store, base) for v in value)
    if isinstance(value, Path):
        return Path(remap(str(value), run, store, base))
    return value


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from strings(v)


def flag(args, name, default=None):
    for i, item in enumerate(args):
        if item == name:
            return args[i+1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return default


def relocated_fit(blob, run, store, base):
    # Only metadata and explicit path values; fitted numerical arrays stay intact.
    seen = set()
    changes = []
    def walk(obj):
        if isinstance(obj, (str, Path)):
            value = remap(obj, run, store, base)
            if value != obj:
                changes.append(True)
            return value
        if id(obj) in seen:
            return obj
        seen.add(id(obj))
        if isinstance(obj, dict):
            for k in list(obj):
                obj[k] = walk(obj[k])
        elif isinstance(obj, list):
            obj[:] = [walk(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(walk(v) for v in obj)
        elif type(obj).__module__.startswith("exp.offline_search") or type(obj).__name__ == "LibraryView":
            for k, v in vars(obj).items():
                setattr(obj, k, walk(v))
        return obj
    return walk(blob), bool(changes)


def build_plan(run, names, work, *, store=STORE, base=BASE):
    run, work = Path(run).resolve(), Path(work)
    rows = {r["arm"]: r for r in json.loads((run / "arms.json").read_text())}
    if not names:
        raise ValueError("explicit arm list required")
    work.mkdir(parents=True, exist_ok=True)
    files, prepared = {}, []

    def add(path, reason, *, destination=None, content=None):
        path = Path(str(path).replace("<RUN>", str(run)))
        if not path.is_file():
            raise FileNotFoundError(f"{reason}: missing dependency {path}")
        target = destination or remap(str(path), run, store, base)
        rel = str(Path(target).relative_to(base))
        if "\n" in rel or ".." in Path(rel).parts:
            raise ValueError("unsafe transfer path")
        if rel in files:
            files[rel]["reasons"].append(reason)
            return
        source_sha = sha(path)
        source = path
        if content is not None:
            source = work / "generated" / rel
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(content)
        files[rel] = dict(rel=rel, source=str(source), original=str(path), source_sha256=source_sha,
                          sha256=sha(source), size=source.stat().st_size, reasons=[reason])

    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError("invalid arm name")
        row = copy.deepcopy(rows[name])
        args = row.get("plugin_args", [])
        if any(x.startswith(("--os-debug", "--os-log-inputs", "--os-oracle", "--trace-")) for x in args):
            raise ValueError(f"{name}: standard topology refuses debug/oracle/trace capture flags")
        if row.get("method") == "exp.offline_search.rounds.r08.abl.clip.method:ClipAWM":
            # CLIP loads a separate image tower on query, after loading its prefit.
            # Its default lives in HOME's HF cache rather than a method kwarg.
            from exp.offline_search.rounds.r08.abl.clip.encoder import WEIGHT_SHA256, weights_path
            weight = Path(row.get("server_env", {}).get("R8_CLIP_WEIGHTS") or weights_path())
            add(weight, name + ": online CLIP image tower")
            entry = files[str(Path(remap(str(weight), run, store, base)).relative_to(base))]
            if entry["source_sha256"] != WEIGHT_SHA256:
                raise ValueError(f"{name}: CLIP image tower SHA mismatch")
            row.setdefault("server_env", {})["R8_CLIP_WEIGHTS"] = str(weight)
        import yaml
        config = yaml.safe_load(Path(row["yaml"]).read_text())
        if config.get("trace") or config.get("debug"):
            raise ValueError(f"{name}: trace/debug configuration forbidden")
        add(row["yaml"], name + ": server config", content=yaml.safe_dump(remap(config, run, store, base), sort_keys=False).encode())
        add(row["matrix"], name + ": client matrix")
        library = config["backend"]["in_memory"]["preload_path"]
        add(library, name + ": native payload PKL")
        if row["mode"] != "stock":
            key = row["cell"].rsplit("_", 1)[0]
            current = store / "library" / key / "current"
            for fname in ("action.npy", "task_id.npy", "ids.json", "manifest.json"):
                path = current / fname
                content = None
                if fname == "manifest.json":
                    content = json.dumps(remap(json.loads(path.read_text()), run, store, base), indent=1).encode()
                add(path, name + ": PluginRuntime current", content=content)
            epj = store / "queries" / row["cell"] / "episodes.json"
            if epj.exists():
                add(epj, name + ": episode identity map")
            if row["mode"] == "plugin":
                artifact = flag(args, "--os-fit-artifact")
                if not artifact:
                    raise ValueError(f"{name}: prefit with --os-fit-artifact required for an exact bounded dependency plan")
                artifact = Path(artifact.replace("<RUN>", str(run)))
                with open(artifact, "rb") as f:
                    blob = pickle.load(f)  # trusted research artifact, no fitting or GPU use
                want = dict(spec=row["method"], kwargs=row.get("kwargs") or {}, cell=row["cell"])
                if any(blob.get(k) != v for k, v in want.items()):
                    raise ValueError(f"{name}: fit metadata differs from arms.json")
                # PluginRuntime enumerates stored library directories and opens task_id/action,
                # even if an arm uses only current. No keys, tokens or images are needed online.
                for directory in sorted((store / "library" / key).iterdir()):
                    if (directory.is_dir() and directory.name != "current" and not directory.name.startswith("_")
                            and "." not in directory.name and (directory / "key_v0.npy").exists()):
                        # library_names uses key_v0's existence to attest a complete library.
                        # Preserve that exact discovery rule; do not synthesize marker arrays.
                        for fname in ("task_id.npy", "action.npy", "key_v0.npy"):
                            add(directory / fname, name + ": runtime payload tables")
                blob, changed = relocated_fit(blob, run, store, base)
                content = pickle.dumps(blob, protocol=4) if changed else None
                add(artifact, name + ": fitted method", content=content)
            # A root flag selects the implicit cell dependencies above, not all 365 GB.
            for value in strings([row.get("kwargs", {}), args, row.get("server_env", {})]):
                value = value.replace("<RUN>", str(run))
                if value in (str(store), "/dev/shm/offline_search_store") or not value.startswith("/"):
                    continue
                path = Path(value)
                if path.is_dir():
                    for child in sorted(path.rglob("*")):
                        if child.is_file():
                            add(child, name + ": explicit directory argument")
                elif path.is_file():
                    add(path, name + ": explicit argument")
                else:
                    raise FileNotFoundError(f"{name}: missing explicit argument {path}")
            if not flag(args, "--os-root"):
                args += ["--os-root", str(store)]
        prepared.append(remap(row, run, store, base))
    data = dict(run=str(run), remote_run=str(base / "runs" / run.name), arms=prepared,
                files=sorted(files.values(), key=lambda x: x["rel"]))
    data["bytes"] = sum(x["size"] for x in data["files"])
    (work / "plan.json").write_text(json.dumps(data, indent=1))
    print(f"PLAN files={len(files)} bytes={data['bytes']} GiB={data['bytes']/(1<<30):.3f}")
    for item in data["files"]:
        print(f"{item['size']:>12} {item['rel']}")
    return data
