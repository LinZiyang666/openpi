"""Small model adapter and startup provenance, without model execution."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from ..schema import SCHEMA_VERSION, status
from .diagnostics import method_chain


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(runtime, config):
    meta = dict(getattr(getattr(runtime, "lib", None), "meta", {}) or {})
    dims = getattr(runtime, "dims", None)
    overrides = config.get("model_manifest", {})
    image_shape = meta.get("img_shape") or {"pi05": [224, 224, 3], "groot": [256, 256, 3]}.get(runtime.model)
    result = dict(model=runtime.model, suite=getattr(runtime, "suite", None), H=runtime.H,
                  camera_names=["third", "wrist"],
                  camera_wire_keys={"third": "observation/image", "wrist": "observation/wrist_image"},
                  image_shapes=meta.get("image_shapes") or ({c: image_shape for c in ("third", "wrist")} if image_shape else {}),
                  state_dim=int(getattr(dims, "RAW_STATE_DIM", meta.get("raw_state_dim", 8))),
                  normalized_state_dim=int(meta.get("Drs", meta.get("rs_valid_dims", 8))),
                  action_dim=int(getattr(dims, "ACT_FULL_DIMS", 32)),
                  valid_action_dims=list(range(int(meta.get("act_valid_dims", 7)))),
                  control_dt=meta.get("control_dt"), block_controls=int(meta.get("exec_steps", 5)),
                  image_shapes_status=status("available" if meta.get("image_shapes") or image_shape else "unsupported",
                                             "wire shapes are also recorded per decision"),
                  control_dt_status=status("available" if meta.get("control_dt") is not None else "unsupported",
                                           "environment control interval is supplied by the client adapter"))
    result.update(overrides)
    if "control_dt" in overrides and "control_dt_status" not in overrides:
        result["control_dt_status"] = status("available" if overrides["control_dt"] is not None else "unsupported")
    default_prices = {"pi05": {"full": .152, "policy": .848, "wrist_only": .0646, "completion": .0502},
                      "groot": {"full": .148, "policy": .852}}
    result["cost_weights"] = config.get("cost_weights", default_prices.get(runtime.model, {}))
    return result


def git_provenance():
    repo = Path(__file__).resolve().parents[4]
    provenance = {}
    for name, command in (("git_head", ["git", "rev-parse", "HEAD"]),
                          ("dirty_diff_sha256", ["git", "diff", "--binary", "HEAD", "--", ".", ":(exclude)tests/review_tests"])):
        try:
            data = subprocess.check_output(command, cwd=str(repo), stderr=subprocess.DEVNULL)
            provenance[name] = data.decode().strip() if name == "git_head" else hashlib.sha256(data).hexdigest()
        except (OSError, subprocess.CalledProcessError) as exc:
            provenance[name] = status("unsupported", str(exc))
    return provenance


def startup_meta(runtime, config, manifest, digest_cache=None):
    artifacts = {}
    paths = {"fit": getattr(runtime.opts, "os_fit_artifact", "")}
    for i, obj in enumerate(method_chain(getattr(runtime, "method", None))):
        source = getattr(sys.modules.get(type(obj).__module__), "__file__", None)
        if source:
            paths["method_{}_code".format(i)] = source
        for name in ("base_fit", "wrist_fit", "stage_fit", "stages_path", "follow_stages_path",
                     "calibration_path", "stall_model_path"):
            value = getattr(obj, name, None)
            if value:
                paths["method_{}_{}".format(i, name)] = value
    server_dir = Path(__file__).resolve().parent
    for source in server_dir.glob("*.py"):
        paths["observer_code_" + source.name] = source
    paths["schema_code"] = server_dir.parent / "schema.py"
    paths["plugin_code"] = server_dir.parent.parent / "closed_loop/plugin.py"
    for lib, table in getattr(runtime, "tables", {}).items():
        # Content hash covers deployed payloads; source array hashes are retained
        # separately when catalog metadata exposes them.
        artifacts["library_" + lib] = (digest_cache.array(table) if digest_cache is not None else
                                      hashlib.sha256(memoryview(table).cast("B")).hexdigest())
        library_dir = getattr(getattr(runtime, "lib", None), "dir", None)
        if library_dir:
            library_dir = Path(library_dir).parent / lib
            manifest_path = library_dir / "manifest.json"
            if manifest_path.exists():
                paths["library_" + lib + "_manifest"] = manifest_path
            for filename in ("key_v0.npy", "key_v1.npy", "rs.npy", "action.npy", "next.npy", "prev.npy",
                             "task_id.npy", "episode.npy", "step.npy", "success.npy"):
                if (library_dir / filename).exists():
                    paths["library_" + lib + "_" + filename] = library_dir / filename
    for name, path in paths.items():
        if path:
            try:
                digest = digest_cache.file(path) if digest_cache is not None else sha_file(path)
                artifacts[name] = {"path": str(path), "sha256": digest, "status": "available"}
            except OSError as exc:
                artifacts[name] = status("error", str(exc))
    provenance = git_provenance()
    return dict(schema=SCHEMA_VERSION, pid=os.getpid(), model_manifest=manifest, **manifest,
                method_spec=getattr(runtime.opts, "os_method", "unknown"),
                method_class=getattr(runtime, "method_name", "unknown"),
                method_kwargs=getattr(runtime.opts, "kwargs", {}),
                library_meta=dict(getattr(getattr(runtime, "lib", None), "meta", {}) or {}),
                method_fit_info=getattr(getattr(runtime, "method", None), "fit_info", {}),
                artifacts=artifacts, capture_config=config, oracle=bool(getattr(runtime, "oracle", False)),
                digest_cache=digest_cache.stats() if digest_cache is not None else status("not_applicable"),
                **provenance)
