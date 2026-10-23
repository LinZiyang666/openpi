"""Library-only bridge to P1 layer math; no training on queries/outcomes."""
from __future__ import annotations

from collections import Counter
from types import MappingProxyType

from exp.r17_integration.debug.validate import digest, episode_identity
from .store import atomic_json


class LibraryView:
    def __init__(self, store):
        if any(row["split"] != "library" for row in store.rows):
            raise ValueError("fit cannot read dev/test store")
        self._store = store
        # Outcomes/status/cost/capture metadata deliberately absent from fit.
        self.rows = tuple(MappingProxyType({k: row[k] for k in (
            "task_name", "task_uid", "attempt", "decision_seq", "orig_init_idx", "task_data_hash")})
                          for row in store.rows)
        self.episode_ids = tuple(digest(episode_identity(row)) for row in store.rows)
        self.fields = tuple(name for name in store.fields if name.startswith("key_") or name in (
            "state", "normalized_actions", "active_state", "active_actions"))

    def array(self, name):
        if name not in self.fields:
            raise KeyError("fit read outside library feature/action allowlist: " + name)
        return self._store.array(name)

    def donor_indices(self, *, exclude_episode):
        if exclude_episode not in self.episode_ids:
            raise ValueError("LOEO exclusion must be a complete source episode")
        return tuple(i for i, ep in enumerate(self.episode_ids) if ep != exclude_episode)


def fit_library(store, *, fitter=None, corrector=False, config=None, receipt_path=None, compact_loeo=False):
    view = LibraryView(store)
    episodes = {ep: row["task_name"] for ep, row in zip(view.episode_ids, view.rows)}
    counts = Counter(episodes.values())
    if corrector and any(n < 3 for n in counts.values()):
        raise ValueError("LOEO corrector needs >=3 complete library episodes per task")
    if fitter is None and corrector:
        raise RuntimeError("P1 corrector fitting hook required; no local replacement math")
    result = fitter(view, config=config or {}, corrector=corrector) if fitter else None
    receipt = dict(version="r17.fit.v1", store_digest=store.manifest["content_digest"],
                   binding=store.manifest["binding"], config=config or {}, episodes_per_task=dict(counts),
                   corrector=corrector, fitter_status="injected" if fitter else "raw_retrieval_only",
                   loeo_exclusions=({'rule':'exclude complete own episode', 'episode_identities':sorted(episodes)}
                       if compact_loeo else {ep: list(view.donor_indices(exclude_episode=ep)) for ep in episodes}))
    receipt["digest"] = digest(receipt)
    if receipt_path is not None:
        atomic_json(receipt_path, receipt)
    return result, receipt


def library_data(store, *, task_names):
    """Construct P1 LibraryData from sealed arrays, never from outcome records."""
    import numpy as np
    from exp.r17_integration.common.layers import LibraryData
    view = LibraryView(store)
    task_names = tuple(task_names)
    if len(task_names) != len(set(task_names)):
        raise ValueError("duplicate canonical task names")
    tasks = {name: i for i, name in enumerate(task_names)}
    if any(row["task_name"] not in tasks for row in view.rows):
        raise ValueError("library task outside canonical benchmark map")
    episodes = {name: i for i, name in enumerate(sorted(set(view.episode_ids)))}
    originals = [row.get('library_episode_id') for row in store.rows]
    if any(value is not None for value in originals):
        if any(type(value) is not int or value < 0 for value in originals):
            raise ValueError('library episode IDs must be complete nonnegative integers')
        pairs = set(zip(view.episode_ids, originals))
        if len(pairs) != len(episodes) or len({value for _,value in pairs}) != len(episodes):
            raise ValueError('library episode IDs must map one-to-one to source episodes')
        episodes = dict(pairs)
    features = {name[4:]: view.array(name) for name in view.fields
                if name.startswith("key_") and name != "key_robot_state"}
    return LibraryData(features=features, state=view.array("state"), action=view.array("normalized_actions"),
                       task_id=np.array([tasks[r["task_name"]] for r in view.rows], np.int64),
                       episode=np.array([episodes[e] for e in view.episode_ids], np.int64),
                       step=np.array([r["decision_seq"] for r in view.rows], np.int64),
                       source_digest=store.manifest["content_digest"])


def fit_recipe(store, *, task_names, corrector=True, receipt_path=None, compact_loeo=False, **settings):
    """Use the published P1 math directly; all five layers remain shared."""
    from exp.r17_integration.common.contracts import ActionStateLayout
    from exp.r17_integration.common.layers import fit_library as shared_fit
    layout = ActionStateLayout(**store.manifest["binding"]["model"]["layout"])
    data = library_data(store, task_names=task_names)
    def fitter(view, *, config, corrector):
        return shared_fit(data, layout=layout, task_count=len(task_names), corrector=corrector, **settings)
    return fit_library(store, fitter=fitter, corrector=corrector,
                       config=dict(task_names=list(task_names), **settings), receipt_path=receipt_path,
                       compact_loeo=compact_loeo)


def save_fitted_recipe(recipe, receipt, directory):
    """Persist immutable P1 fitted arrays; do not serialize mutable episode memo."""
    import numpy as np
    from pathlib import Path
    from types import SimpleNamespace
    from exp.offline_search.debug.schema import write_npz_block
    from exp.r17_integration.debug.validate import file_sha, plain
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {}
    def encode(value):
        if isinstance(value, np.ndarray):
            key = f"array_{len(arrays):05d}"
            arrays[key] = value
            return {"array": key, "order": "F" if value.flags.f_contiguous and not value.flags.c_contiguous else "C"}
        if isinstance(value, SimpleNamespace):
            return {"namespace": encode(vars(value))}
        if isinstance(value, dict):
            return {"mapping": [[plain(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"sequence": [encode(v) for v in value], "tuple": isinstance(value, tuple)}
        return {"value": plain(value)}
    fields = ("action", "state", "task", "episode", "step", "fields", "pca", "projected", "sigma", "tasks", "heads")
    fitted = {key: getattr(recipe, key) for key in fields}
    if hasattr(recipe, "distance_scale"):
        fitted["distance_scale"] = recipe.distance_scale
    tree = encode(fitted)
    write_npz_block(directory / "fit.npz", arrays)
    manifest = dict(schema="r17.fitted_recipe.v1", receipt=receipt, tree=tree, fit_evidence=recipe.fit_evidence,
                    arrays_sha256=file_sha(directory / "fit.npz"))
    manifest["digest"] = digest(manifest)
    atomic_json(directory / "READY.json", manifest)
    return manifest


def load_fitted_recipe(directory, *, expected_binding):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from exp.offline_search.debug.schema import read_npz_block
    from exp.r17_integration.debug.validate import require_binding
    from exp.r17_integration.common.contracts import ActionStateLayout
    from exp.r17_integration.common.layers import make_method
    directory = Path(directory)
    manifest = json.loads((directory / "READY.json").read_text())
    if manifest["schema"] != "r17.fitted_recipe.v1":
        raise ValueError("fit manifest mismatch")
    require_binding(manifest["receipt"]["binding"], expected_binding)
    arrays = read_npz_block(directory / "fit.npz")
    for array in arrays.values():
        array.flags.writeable = False
    def decode(node):
        if "array" in node:
            import numpy as np
            result = np.array(arrays[node["array"]], copy=True, order=node.get("order", "C"))
            result.flags.writeable = False
            return result
        if "namespace" in node:
            return SimpleNamespace(**decode(node["namespace"]))
        if "mapping" in node:
            return {key: decode(value) for key, value in node["mapping"]}
        if "sequence" in node:
            result = [decode(value) for value in node["sequence"]]
            return tuple(result) if node["tuple"] else result
        return node["value"]
    receipt = manifest["receipt"]
    settings = dict(receipt["config"])
    task_names = settings.pop("task_names")
    settings.pop("guard", None)  # sealed fits recorded the retired guard flag; it is never served
    recipe = make_method(layout=ActionStateLayout(**expected_binding["model"]["layout"]),
                         task_count=len(task_names), corrector=receipt["corrector"], **settings)
    for key, value in decode(manifest["tree"]).items():
        setattr(recipe, key, value)
    recipe.fit_evidence = manifest["fit_evidence"]
    return recipe
