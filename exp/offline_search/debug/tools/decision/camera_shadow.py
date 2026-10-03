"""Wrist/third-only versus full two-camera fixed-fit retrieval agreement."""
import numpy as np

from . import common as C
from .kernels import catalog_map, kernel, overlap
from .metrics import action_metrics


def _single_library(arm, args=None):
    """Compare offline camera proposals by stage without claiming one-camera serving support."""
    decisions, episodes = C.inputs(arm)
    ids = decisions.decision_id.tolist()
    full = C.augmentation(arm, "shadow_look", ids)
    cameras = C.augmentation(arm, "camera_shadow", ids)
    base = C.pick(full, "cache_chunk", "chunk")
    if base is None or not cameras:
        raise C.Unavailable("full shadow_look or camera_shadow unavailable")
    try:
        catalog = catalog_map(arm)
    except C.Unavailable:
        catalog = None
    output = []
    for camera in ("wrist", "third"):
        candidate = C.pick(cameras, camera + "_cache_chunk", camera + "_chunk", "cache_chunk_" + camera,
                           camera + "_only_chunk", "chunk_" + camera)
        if candidate is None:
            continue
        crows = C.pick(cameras, camera + "_rows", "rows_" + camera)
        cweights = C.pick(cameras, camera + "_weights", "weights_" + camera)
        cscores = C.pick(cameras, camera + "_scores", "scores_" + camera)
        for i, row in enumerate(decisions.to_dict("records")):
            metrics = action_metrics(arm, candidate[i], base[i])
            record = dict(C.identity(row), camera=camera, offset_denominator=len(metrics),
                          status="available" if metrics else "unavailable",
                          lib=str(full["lib"][i]) if "lib" in full and str(full["lib"][i]) else C.field(row, "lib"))
            if metrics:
                record.update(rms=float(np.sqrt(np.mean([m["mse"] for m in metrics]))), units=metrics[0]["units"],
                    gripper_flip=float(np.mean([m["gripper_flip"] for m in metrics])) if metrics[0]["gripper_flip"] is not None else None)
            if all(value is not None for value in (crows, cweights)) and "rows" in full and "weights" in full:
                if all(np.isfinite(value[i]).all() for value in (crows, cweights, full["rows"], full["weights"])):
                    record["kernel_overlap"] = overlap(crows[i], cweights[i], full["rows"][i], full["weights"][i])
                    if catalog:
                        a = kernel(crows[i], cweights[i], catalog, record["lib"])
                        b = kernel(full["rows"][i], full["weights"][i], catalog, record["lib"])
                        record["stage_mass_l1"] = float(sum(abs(a["stage_mass"].get(key, 0) - b["stage_mass"].get(key, 0))
                                                  for key in set(a["stage_mass"]) | set(b["stage_mass"])))
                        record["top_demo_same"] = a["top_demo"] == b["top_demo"] if a["top_demo"] and b["top_demo"] else None
                        def stage_class(info):
                            positive = {key: value for key, value in info["stage_mass"].items() if value > 0}
                            if not positive:
                                return "unknown"
                            if info["unknown_stage_mass"] > 0 or len(positive) > 1:
                                return "mixed"
                            return "stage:" + next(iter(positive))
                        record["stage_class_same"] = stage_class(a) == stage_class(b)
            if cscores is not None and "scores" in full:
                den = C.number(full["scores"][i][0])
                num = C.number(cscores[i][0])
                record["first_score_ratio"] = num / den if num is not None and den is not None and den != 0 else None
            output.append(record)
    available = sum(row["status"] == "available" for row in output)
    return {"status": "available" if available else "unavailable", "coverage": C.coverage(2 * len(decisions), available),
            "episode_denominator": len(episodes), "notes": ["Denominator is two camera alternatives per decision.",
               "Score ratios retain the fit's metric units; no unvalidated admission threshold is inferred."],
            "tables": {"decisions": output, "by_stage": C.summarize(output, ["camera", "stage", "lib"],
                ["rms", "gripper_flip", "kernel_overlap", "stage_mass_l1", "stage_class_same", "top_demo_same", "first_score_ratio"])}}


def analyze(arm, args=None):
    """Compare camera retrievals for every augmented library, including pure-policy banks."""
    decisions, _ = C.inputs(arm)
    full = C.augmentation(arm, "shadow_look", decisions.decision_id.tolist())
    suffixes = [""] + sorted(key[len("cache_chunk"):] for key in full if key.startswith("cache_chunk_"))
    result = _single_library(arm, args)
    if len(suffixes) == 1:
        return result
    class LibraryView:
        def __init__(self, source, suffix):
            self.source, self.suffix = source, suffix
        def __getattr__(self, key):
            return getattr(self.source, key)
        def catalog(self):
            # Reader.catalog is the deployed bank. A second pure-policy bank
            # needs its own catalogue; never label its rows with the first bank.
            return None
        def aug(self, kind, decision_ids=None):
            data = self.source.aug(kind, decision_ids)
            if kind == "shadow_look":
                output = {key[:-len(self.suffix)]: value for key, value in data.items() if key.endswith(self.suffix)}
            elif kind == "camera_shadow":
                output = {key[:-len(self.suffix)]: value for key, value in data.items() if key.endswith(self.suffix)}
            else:
                return data
            if "decision_id" in data:
                output["decision_id"] = data["decision_id"]
            return output
    for suffix in suffixes[1:]:
        try:
            other = _single_library(LibraryView(arm, suffix), args)
        except C.Unavailable as error:
            other = {"coverage": C.coverage(2 * len(decisions), 0), "tables": {"decisions": [
                dict(C.identity(row), camera=camera, lib=suffix[1:], status="unavailable", reason=str(error))
                for row in decisions.to_dict("records") for camera in ("wrist", "third")]}}
        for key in ("denominator", "available", "unavailable"):
            result["coverage"][key] += other["coverage"][key]
        for name, rows in other["tables"].items():
            result["tables"][name].extend(rows)
    result["coverage"]["status"] = "available" if result["coverage"]["available"] else "unavailable"
    result["status"] = result["coverage"]["status"]
    result["notes"].append("Coverage counts both camera alternatives for every augmented library.")
    return result


def main():
    C.cli("camera_shadow", analyze)


if __name__ == "__main__":
    main()
