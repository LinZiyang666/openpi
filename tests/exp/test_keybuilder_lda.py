"""Tests for the key-builder x LDA supplement (emitter, strict summary entry, fit checks, launcher).

Covers the edges that would contaminate the reported table or make the
admission checks hollow:

* emitter: three-field templates need zscore+tanh on every field; a fit from
  another template/library or a degenerate/off-simplex fit is refused; a
  zero-weight field drops its normalizer but the builder's required fields stay
  enabled; the LLM arm carries the real ``cp1_llm_layer_extract`` fields with
  ``robot_state`` at weight 0; unknown yaml keys fail acceptance; CLIP deferral
  yields exactly 3 + 1 arms per suite; yaml_ids are unique across suites.
* strict summary: a whole missing arm, one missing episode, a foreign arm in a
  run journal, a suite/A-pool mismatch in the launch record, an accepted retry
  whose per-step rows sit at a stale attempt, a worker error on an accepted
  row, cross-file duplicates (identical ok, conflicting not), a non-FULL_HIT
  step.
* fit script: template/progress/degeneracy checks.
* launcher: with stubbed ``tmux`` / ``pkill`` / ``ss`` / interpreter, the two
  Stage-2 devices and matrix/boot paths are forwarded, relative paths and bad
  devices exit before anything launches, and the defaults are unchanged. No
  GPU, server or real process is touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from openpi.cache.config import ConfigValidationError, load_cache_config

import exp.weighted_sum.emit_keybuilder_lda as E
from exp.weighted_sum.analysis import lcw_ablation_summary as S

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "exp/weighted_sum/analysis"))
import lcw_fit_weights as F  # noqa: E402  (sibling import inside the analysis dir)

SUITES = E.SUITES
DIMS = {
    "cp1_mean_pool": {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32},
    "cp1_max_pool": {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32},
    "cp1_spatial_pool_64": {"vision_0": 8192, "vision_1": 8192, "prompt_emb": 2048, "robot_state": 32},
    "clip_vit_b_32": {"vision_0": 512, "vision_1": 512, "prompt_emb": 2048, "robot_state": 32},
    E.LLM_STEM: {"vision_0": 2048, "robot_state": 32},
}
ROLLUP = {s: hashlib.sha256(s.encode()).hexdigest() for s in SUITES}


# ------------------------------------------------------------------
# Fixture helpers
# ------------------------------------------------------------------
def calib_entry(stem: str, *, method: str = "zscore", squash: str = "tanh", drop: str | None = None,
                sigma: float = 0.01) -> dict:
    fields = ["vision_0", "robot_state"] if stem == E.LLM_STEM else list(E.FIELDS)
    if drop:
        fields.remove(drop)
    out = {}
    for f in fields:
        l2 = f == "robot_state"
        out[f] = {"sim_type": "l2" if l2 else "cosine", "shortlist": [],
                  "selected": {"method": method,
                               "params": {"mu": -1.8 if l2 else 0.95, "sigma": 1.0 if l2 else sigma, "squash": squash}}}
    return {"builder_type": E.BUILDER_OF[stem], "vector_dims": DIMS[stem], "fields": out}


def write_library(path: Path, stem: str, *, clip_variant: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    art = {"key_builder_type": E.BUILDER_OF[stem], "checkpoint_id": "CP1", "vector_dims": DIMS[stem],
           "entries": [types.SimpleNamespace(trajectory_id=f"ep{i // 2}", step_idx=i % 2) for i in range(6)]}
    if stem == E.CLIP_STEM:
        art.update(clip_variant or E.CLIP_VARIANT)
    with path.open("wb") as f:
        pickle.dump(art, f)


def write_fit(path: Path, *, library: Path, template: Path, w, raw=None, queries: int = 100,
              d=None, cov=None, lda_raw=None, template_sha: str | None = None,
              library_sha: str | None = None, drop: tuple[str, ...] = ()) -> None:
    """A fit file as ``lcw_fit_weights.py`` writes it; ``w`` defaults to the clipped, normalised ``raw``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = list(w) if raw is None else raw
    fit = {
        "library": str(library),
        "library_sha256": library_sha or E.sha256_of(library),
        "template": {"path": str(template), "sha256": template_sha or E.sha256_of(template)},
        "fits": {
            E.LDA_FIT: {"raw": raw, "w": list(w)},
            E.LDA_DETAIL: {"fisher_lda": {"d": d or [0.1, 0.2, 0.3],
                                          "cov": cov or [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                          "lda_raw": lda_raw or raw, "queries": queries}},
        },
    }
    for key in drop:
        fit.pop(key)
    path.write_text(json.dumps(fit))


class World:
    """A self-contained tree: libraries, supplement calibrations, grid summary, apool records."""

    def __init__(self, tmp: Path, monkeypatch) -> None:
        self.root = tmp
        self.local = tmp / "art"
        self.data = tmp / "data"
        self.cfg = tmp / "cfg"
        self.served = "/srv/art"
        for suite in SUITES:
            for stem in (*E.POOL_STEMS, E.LLM_STEM):
                write_library(E.local_library(suite, stem, self.local), stem)
                self.set_calibration(suite, stem, calib_entry(stem))
            apool = tmp / f"apool_{suite}.yaml"
            apool.write_text(yaml.safe_dump({"suite": suite, "rollup_sha256": ROLLUP[suite]}))
        monkeypatch.setattr(E, "APOOL_RECORD", str(tmp / "apool_{suite}.yaml"))
        self.grid6 = tmp / "summary_grid6.json"
        cells = {"0/0/6": 0.60, "2/2/2": 0.70, "1/4/1": 0.66, "6/0/0": 0.5}
        self.grid6.write_text(json.dumps({f"pi05_{s}": {"cells": cells} for s in SUITES}))

    def set_calibration(self, suite: str, stem: str, entry: dict) -> None:
        p = self.data / "calibration" / suite / f"{stem}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({stem: entry}))

    def args(self, *extra: str):
        return E.build_parser().parse_args([
            "--cfg", str(self.cfg), "--data", str(self.data), "--local-root", str(self.local),
            "--served-root", self.served, *extra])

    def templates(self) -> None:
        E.run_templates(self.args("templates", "--grid6-summary", str(self.grid6)))

    def template(self, suite: str, stem: str) -> Path:
        return self.cfg / suite / "templates" / f"{stem}_template.yaml"

    def fit_path(self, suite: str, stem: str) -> Path:
        return self.data / "fits" / suite / stem / "lcw_fit_weights.json"

    def fit_all(self, w=(0.5, 0.3, 0.2), **kw) -> None:
        for suite in SUITES:
            for stem in E.POOL_STEMS:
                write_fit(self.fit_path(suite, stem), library=E.local_library(suite, stem, self.local),
                          template=self.template(suite, stem), w=w, **kw)

    def final(self, *extra: str) -> None:
        E.run_final(self.args("final", *extra))

    def yaml(self, suite: str, stem: str) -> Path:
        return self.cfg / suite / f"{E.yaml_id(suite, stem)}.yaml"


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    return World(tmp_path, monkeypatch)


# ------------------------------------------------------------------
# Emitter: templates
# ------------------------------------------------------------------
def test_templates_three_fields_at_one_third_and_diagnostic_cells(world, capsys):
    world.templates()
    for suite in SUITES:
        cells = json.loads((world.data / "diagnostic_cells" / f"{suite}.json").read_text())
        assert len(cells["cells"]) == 4 and all(c["n"] == 500 for c in cells["cells"])
        assert [c for c in cells["cells"] if c["w"] == [1 / 3, 1 / 3, 1 / 3]][0]["sr"] == 0.70
        assert "diagnostic only" in cells["note"]
        for stem in E.POOL_STEMS:
            doc = yaml.safe_load(world.template(suite, stem).read_text())
            assert {f: doc["keys"][f]["weight"] for f in E.FIELDS} == {f: pytest.approx(1 / 3) for f in E.FIELDS}
            assert set(doc["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]) == set(E.FIELDS)
            assert doc["backend"]["in_memory"]["preload_path"] == E.served_library(suite, stem, world.served)
    out = capsys.readouterr().out
    assert out.count("lcw_fit_weights.py --library") == 8


def test_template_refuses_non_zscore_and_missing_field(world):
    world.set_calibration("libero_10", "cp1_max_pool", calib_entry("cp1_max_pool", method="logit"))
    with pytest.raises(SystemExit, match="assumes zscore"):
        world.templates()
    world.set_calibration("libero_10", "cp1_max_pool", calib_entry("cp1_max_pool", drop="vision_1"))
    with pytest.raises(SystemExit, match="no calibration"):
        world.templates()
    world.set_calibration("libero_10", "cp1_max_pool", calib_entry("cp1_max_pool", sigma=0.0))
    with pytest.raises(SystemExit, match="unusable mu/sigma"):
        world.templates()


def test_base_calibration_is_used_when_no_supplement(world, tmp_path):
    base = tmp_path / "base_{suite}.json"
    (tmp_path / "base_libero_spatial.json").write_text(json.dumps({"cp1_mean_pool": calib_entry("cp1_mean_pool")}))
    entry, path = E.resolve_calibration("libero_spatial", "cp1_mean_pool", supplement_dir=tmp_path / "none",
                                        base=str(base))
    assert path == tmp_path / "base_libero_spatial.json" and entry["builder_type"] == "cp1_mean_pool"
    with pytest.raises(SystemExit, match="no calibration"):
        E.resolve_calibration("libero_spatial", "cp1_max_pool", supplement_dir=tmp_path / "none", base=str(base))


# ------------------------------------------------------------------
# Emitter: final arms
# ------------------------------------------------------------------
def test_final_writes_arms_matrices_weights_manifest(world):
    world.templates()
    world.fit_all(w=(0.5, 0.3, 0.2))
    world.final()
    for suite in SUITES:
        for stem in E.POOL_STEMS:
            cfg = load_cache_config(world.yaml(suite, stem))
            assert (cfg.keys.vision_0.weight, cfg.keys.vision_1.weight, cfg.keys.robot_state.weight) == (0.5, 0.3, 0.2)
            tpl = E.normalizers_of(yaml.safe_load(world.template(suite, stem).read_text()))
            assert dict(cfg.checkpoints["cp1"].search_strategy.score_normalization.fields) == tpl
            assert cfg.key_builder.type == E.BUILDER_OF[stem]
        pool = yaml.safe_load((world.cfg / suite / f"matrix_{suite}_pool.yaml").read_text())
        llm = yaml.safe_load((world.cfg / suite / f"matrix_{suite}_llm.yaml").read_text())
        assert pool["suite"] == suite and [r["arm"] for r in pool["arms"]] == [E.yaml_id(suite, s) for s in E.POOL_STEMS]
        assert [r["arm"] for r in llm["arms"]] == [E.yaml_id(suite, E.LLM_STEM)]
        assert all(r["sidecar"] is None and Path(r["yaml"]).stem == r["arm"] for r in pool["arms"] + llm["arms"])
    weights = json.loads((world.data / "weights.json").read_text())
    rec = weights[E.yaml_id("libero_10", "cp1_mean_pool")]
    assert rec["lda"]["w"] == [0.5, 0.3, 0.2] and rec["lda"]["queries"] == 100
    assert rec["library_sha256"] == E.sha256_of(E.local_library("libero_10", "cp1_mean_pool", world.local))
    assert rec["template_sha256"] == E.sha256_of(world.template("libero_10", "cp1_mean_pool"))
    assert set(rec["selected"]) == set(E.FIELDS)
    manifest = json.loads((world.data / "active_manifest.json").read_text())
    spec = manifest["suites"]["libero_10"]
    assert spec["apool_rollup_sha256"] == ROLLUP["libero_10"]
    assert spec["rounds"]["pool"]["arms"] == [E.yaml_id("libero_10", s) for s in E.POOL_STEMS]
    assert spec["rounds"]["llm"]["arms"] == [E.yaml_id("libero_10", E.LLM_STEM)]
    assert spec["reference"]["arms"] == ["fw_pi05_libero_10_lda"] and spec["reference"]["launch_binding"] == "legacy"
    for rnd in ("pool", "llm"):
        assert spec["rounds"][rnd]["launch"] == [] and spec["rounds"][rnd]["launch_binding"] == "run_id"
        assert spec["rounds"][rnd]["launch_glob"].endswith("per_step.jsonl.launch.*.json")
    assert manifest["clip_deferred"] is False and manifest["episodes_per_arm"] == 500
    ids = [a for s in manifest["suites"].values() for r in s["rounds"].values() for a in r["arms"]]
    assert len(ids) == 10 == len(set(ids))


@pytest.mark.parametrize("bad", [
    dict(w=(0.5, 0.5, 0.5)),                                    # off the simplex
    dict(w=(0.0, 0.0, 0.0)),                                    # no mass
    dict(w=(0.5, 0.3, 0.2), raw=[float("nan"), 1, 1]),          # non-finite raw
    dict(w=(0.5, 0.3, 0.2), queries=0),                          # empty population
    dict(w=(0.5, 0.3, 0.2), d=[float("inf"), 0, 0]),             # non-finite intermediates
    dict(w=(1 / 3, 1 / 3, 1 / 3), raw=[-1.0, -2.0, -3.0]),       # hand-filled uniform over a negative direction (N1)
    dict(w=(0.5, 0.3, 0.2), raw=[5.0, 3.0, 2.0], lda_raw=[5.0, 3.0, 1.0]),  # raw disagrees with the detail (N1)
    dict(w=(0.5, 0.3, 0.2), drop=("library_sha256",)),          # no library identity (B2)
    dict(w=(0.5, 0.3, 0.2), drop=("template",)),                # no template identity (B2)
    dict(w=(0.5, 0.3, 0.2), library_sha="0" * 64),              # library replaced since the fit (B2)
])
def test_final_refuses_degenerate_fit(world, bad):
    world.templates()
    world.fit_all()
    write_fit(world.fit_path("libero_spatial", "cp1_max_pool"),
              library=E.local_library("libero_spatial", "cp1_max_pool", world.local),
              template=world.template("libero_spatial", "cp1_max_pool"), **bad)
    with pytest.raises(SystemExit):
        world.final()


def test_final_refuses_fit_from_other_template_or_library(world):
    world.templates()
    world.fit_all()
    suite, stem = "libero_spatial", "cp1_mean_pool"
    # The library rebuilt at the same path after the fit.
    write_fit(world.fit_path(suite, stem), library=E.local_library(suite, stem, world.local),
              template=world.template(suite, stem), w=(0.5, 0.3, 0.2))
    write_library(E.local_library(suite, stem, world.local), stem)
    with open(E.local_library(suite, stem, world.local), "ab") as f:
        f.write(b"rebuilt")
    with pytest.raises(SystemExit, match="changed since the fit"):
        world.final()
    write_library(E.local_library(suite, stem, world.local), stem)
    write_fit(world.fit_path(suite, stem), library=E.local_library(suite, stem, world.local),
              template=world.template(suite, stem), w=(0.5, 0.3, 0.2), template_sha="0" * 64)
    with pytest.raises(SystemExit, match="changed since the fit"):
        world.final()
    write_fit(world.fit_path(suite, stem), library=E.local_library(suite, "cp1_max_pool", world.local),
              template=world.template(suite, stem), w=(0.5, 0.3, 0.2))
    with pytest.raises(SystemExit, match="fitted on"):
        world.final()
    world.fit_path(suite, stem).unlink()
    with pytest.raises(SystemExit, match="missing fit"):
        world.final()


def test_final_refuses_calibration_changed_after_template(world):
    world.templates()
    world.fit_all()
    entry = calib_entry("cp1_spatial_pool_64")
    entry["fields"]["vision_1"]["selected"]["params"]["mu"] = 0.5
    world.set_calibration("libero_10", "cp1_spatial_pool_64", entry)
    with pytest.raises(SystemExit, match="differs from the fit template"):
        world.final()


@pytest.mark.parametrize("w, enabled, scored", [
    ((0.7, 0.3, 0.0), {"vision_0": 0.7, "vision_1": 0.3, "robot_state": 0.0}, {"vision_0", "vision_1"}),
    ((0.0, 0.6, 0.4), {"vision_0": 0.0, "vision_1": 0.6, "robot_state": 0.4}, {"vision_1", "robot_state"}),
    ((0.0, 0.0, 1.0), {"vision_0": 0.0, "robot_state": 1.0}, {"robot_state"}),
])
def test_zero_weight_field_keeps_required_fields_without_normalizer(world, w, enabled, scored):
    world.templates()
    world.fit_all(w=w)
    world.final()
    cfg = load_cache_config(world.yaml("libero_spatial", "cp1_mean_pool"))
    served = {n: getattr(cfg.keys, n).weight for n in ("vision_0", "vision_1", "vision_2", "prompt_emb", "robot_state")
              if getattr(cfg.keys, n).enabled}
    assert served == enabled
    assert set(cfg.checkpoints["cp1"].search_strategy.score_normalization.fields) == scored


def test_llm_arm_real_fields_and_zero_weight_state(world):
    world.templates()
    world.fit_all()
    world.final()
    for suite in SUITES:
        path = world.yaml(suite, E.LLM_STEM)
        doc = yaml.safe_load(path.read_text())
        assert doc["key_builder"] == {"type": "cp1_llm_layer_extract", "extract_layer": 0,
                                      "prefix_reducer": {"type": "prefix_mean_pool"}}
        assert "llm_layer" not in doc["key_builder"]
        cfg = load_cache_config(path)
        assert cfg.key_builder.extract_layer == 0 and cfg.key_builder.prefix_reducer.type == "prefix_mean_pool"
        assert (cfg.keys.vision_0.enabled, cfg.keys.vision_0.weight) == (True, 1.0)
        assert (cfg.keys.robot_state.enabled, cfg.keys.robot_state.weight) == (True, 0.0)
        assert not (cfg.keys.vision_1.enabled or cfg.keys.vision_2.enabled or cfg.keys.prompt_emb.enabled)
        assert dict(cfg.backend.vector_dims) == {"vision_0": 2048, "robot_state": 32}
        assert set(cfg.checkpoints["cp1"].search_strategy.score_normalization.fields) == {"vision_0"}
        assert cfg.backend.in_memory.preload_path == E.served_library(suite, E.LLM_STEM, world.served)
    rec = json.loads((world.data / "weights.json").read_text())[E.yaml_id("libero_10", E.LLM_STEM)]
    assert rec["w"] == {"vision_0": 1.0, "robot_state": 0.0} and "lda" not in rec


def test_llm_arm_refuses_wrong_dims(world):
    world.templates()
    world.fit_all()
    entry = calib_entry(E.LLM_STEM)
    entry["vector_dims"] = {"vision_0": 4096, "robot_state": 32}
    world.set_calibration("libero_spatial", E.LLM_STEM, entry)
    with pytest.raises(SystemExit, match="vector_dims"):
        world.final()


def test_skip_clip_yields_three_plus_one_per_suite(world):
    world.templates()
    world.fit_all()
    world.final("--skip-clip", "--skip-clip-reason", "preflight OOM at W=6")
    manifest = json.loads((world.data / "active_manifest.json").read_text())
    assert manifest["clip_deferred"] is True and manifest["clip_deferred_reason"] == "preflight OOM at W=6"
    for suite in SUITES:
        pool = yaml.safe_load((world.cfg / suite / f"matrix_{suite}_pool.yaml").read_text())
        assert [r["arm"] for r in pool["arms"]] == [E.yaml_id(suite, s) for s in E.POOL_STEMS if s != E.CLIP_STEM]
        assert manifest["suites"][suite]["rounds"]["pool"]["arms"] == [r["arm"] for r in pool["arms"]]
        assert E.CLIP_STEM not in manifest["suites"][suite]["libraries"]
        assert not world.yaml(suite, E.CLIP_STEM).exists()
    ids = [a for s in manifest["suites"].values() for r in s["rounds"].values() for a in r["arms"]]
    assert len(ids) == 8 == len(set(ids))


def test_skip_clip_needs_a_reason(world, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--cfg", str(world.cfg), "--data", str(world.data), "final", "--skip-clip"])
    with pytest.raises(SystemExit, match="skip-clip-reason"):
        E.main()


def test_clip_variant_must_match_online_default(world):
    world.templates()
    write_library(E.local_library("libero_10", E.CLIP_STEM, world.local), E.CLIP_STEM,
                  clip_variant={"clip_model_name": "ViT-L-14", "clip_pretrained": "openai"})
    world.fit_all()
    with pytest.raises(SystemExit, match="CLIP variant"):
        world.final()


def test_accept_yaml_flags_unknown_keys_and_recipe_drift(world):
    world.templates()
    world.fit_all()
    world.final()
    path = world.yaml("libero_10", E.LLM_STEM)
    good = yaml.safe_load(path.read_text())
    norm = E.normalizers_of(good)
    kwargs = dict(builder_type="cp1_llm_layer_extract", weights={"vision_0": 1.0}, normalizers=norm,
                  vector_dims={"vision_0": 2048, "robot_state": 32},
                  preload_path=good["backend"]["in_memory"]["preload_path"],
                  key_builder_extra={"extract_layer": 0, "prefix_reducer": {"type": "prefix_mean_pool"}})
    E.accept_yaml(path, **kwargs)
    bad = json.loads(json.dumps(good))
    bad["key_builder"]["llm_layer"] = 0
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit, match="unknown keys"):
        E.accept_yaml(path, **kwargs)
    bad = json.loads(json.dumps(good))
    bad["checkpoints"]["cp1"]["gate"] = {"type": "always_skip"}
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit, match="pure-cache recipe"):
        E.accept_yaml(path, **kwargs)
    # Weight drift: a scored field without a normalizer is already refused by the
    # production loader; a scored field with one is refused by the acceptance.
    bad = json.loads(json.dumps(good))
    bad["keys"]["robot_state"]["weight"] = 0.5
    bad["keys"]["vision_0"]["weight"] = 0.5
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigValidationError, match="missing entries"):
        E.accept_yaml(path, **kwargs)
    bad["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]["robot_state"] = {
        "method": "zscore", "params": {"mu": -1.8, "sigma": 1.0, "squash": "tanh"}}
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit, match="served weights"):
        E.accept_yaml(path, **kwargs)


def test_yaml_ids_carry_suite_and_lda_suffix():
    assert E.yaml_id("libero_spatial", "cp1_mean_pool") == "kb_libero_spatial_cp1_mean_pool_lda"
    assert E.yaml_id("libero_10", E.LLM_STEM) == "kb_libero_10_cp1_llm_l0_prefix_mean_pool"
    assert E.library_rel("libero_10", E.LLM_STEM) == "libero_10/llm_layer_extract/cp1_llm_l0_prefix_mean_pool.pkl"


# ------------------------------------------------------------------
# Strict summary entry
# ------------------------------------------------------------------
def _uid(arm: str, t: int, e: int, phase: str = "eval") -> str:
    return f"{arm}:{phase}:{t}:{e}"


class Run:
    """Writes one round's journal / per_step / launch for a set of arms (10 x 50 episodes each).

    The launch record follows the driver's contract: ``run_id`` on it, on
    every journal row and on every per-step row. ``legacy=True`` writes the
    pre-run-id record the historical reference arm has.
    """

    def __init__(self, directory: Path, suite: str, arms: list[str], *, rollup: str | None = None,
                 smoke: bool = False, launch_arms: list[str] | None = None, run_id: str = "run0",
                 legacy: bool = False) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.suite = suite
        self.arms = arms
        self.run_id = run_id
        self.rows: list[dict] = []
        self.steps: list[dict] = []
        for arm in arms:
            for t in range(10):
                for e in range(50):
                    self.add(arm, t, e, success=(t + e) % 3 == 0)
        self.launch = {"suite": suite, "arms": sorted(launch_arms if launch_arms is not None else arms),
                       "trials_per_task": 50, "smoke": smoke,
                       "apool": {"suite": suite, "rollup_sha256": rollup or ROLLUP[suite]}}
        if not legacy:
            self.launch["run_id"] = run_id
        self.launch_path = self.dir / ("per_step.jsonl.launch.json" if legacy else f"per_step.jsonl.launch.{run_id}.json")
        self.write_launch()

    def write_launch(self) -> None:
        self.launch_path.write_text(json.dumps(self.launch))

    def add(self, arm: str, t: int, e: int, *, success: bool, attempt: int = 1, accepted: bool = True,
            error: str | None = None, run_id: str | None = None, steps: tuple[str, ...] = ("FULL_HIT", "FULL_HIT"),
            step_attempt: int | None = None, step_run_id: str | None = None, step_accepted: bool | None = None,
            phase: str = "eval", row_phase: str | None = None, row_arm: str | None = None) -> None:
        run_id = self.run_id if run_id is None else run_id
        row = {"task_uid": _uid(arm, t, e, phase), "yaml_id": row_arm or arm, "phase": row_phase or phase,
               "status": "done" if success else "failed", "success": success, "ts": 1.0,
               "attempt": attempt, "accepted": accepted, "run_id": run_id}
        if error:
            row["error"] = error
        self.rows.append(row)
        for i, hit in enumerate(steps):
            self.steps.append({"yaml_id": row_arm or arm, "task_uid": _uid(arm, t, e, phase), "step_idx": i * 5,
                               "hit_type": hit, "attempt": attempt if step_attempt is None else step_attempt,
                               "accepted": accepted if step_accepted is None else step_accepted,
                               "run_id": run_id if step_run_id is None else step_run_id})

    def drop(self, arm: str, t: int, e: int) -> None:
        uid = _uid(arm, t, e)
        self.rows = [r for r in self.rows if r["task_uid"] != uid]
        self.steps = [r for r in self.steps if r["task_uid"] != uid]

    def write(self, name: str = "") -> tuple[Path, Path]:
        j = self.dir / f"journal{name}.jsonl"
        p = self.dir / f"per_step{name}.jsonl"
        j.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        p.write_text("".join(json.dumps(r) + "\n" for r in self.steps))
        return j, p


def make_manifest(tmp: Path, suite: str = "libero_10", *, pool_arms=None, llm_arms=None, ref_extra=True,
                  use_glob: bool = False):
    pool_arms = pool_arms or [E.yaml_id(suite, s) for s in E.POOL_STEMS]
    llm_arms = llm_arms or [E.yaml_id(suite, E.LLM_STEM)]
    ref_arm = f"fw_pi05_{suite}_lda"
    ref_all = [ref_arm] + ([f"fw_pi05_{suite}_{a}" for a in ("uniform", "acterr", "leader")] if ref_extra else [])
    runs = {"pool": Run(tmp / "pool", suite, pool_arms, run_id="runpool"),
            "llm": Run(tmp / "llm", suite, llm_arms, run_id="runllm"),
            "reference": Run(tmp / "ref", suite, ref_all, run_id="runref", legacy=True)}
    spec = {"apool_rollup_sha256": ROLLUP[suite], "rounds": {}, "reference": {}}

    def finish():
        for name, run in runs.items():
            j, p = run.write()
            arms = llm_arms if name == "llm" else pool_arms if name == "pool" else [ref_arm]
            entry = {"arms": arms, "journal": str(j), "per_step": str(p)}
            if name == "reference":
                entry.update({"launch": str(run.launch_path), "launch_binding": "legacy"})
                spec["reference"] = entry
            else:
                entry["launch_binding"] = "run_id"
                if use_glob:
                    entry.update({"launch": [], "launch_glob": str(run.dir / "per_step.jsonl.launch.*.json")})
                else:
                    entry["launch"] = str(run.launch_path)
                spec["rounds"][name] = entry
        return spec
    return runs, finish


def test_manifest_merge_happy_path_and_reference_selection(tmp_path):
    runs, finish = make_manifest(tmp_path)
    spec = finish()
    arms, prov = S.load_manifest_suite("libero_10", spec, expected=500)
    assert set(arms) == set(spec["rounds"]["pool"]["arms"]) | set(spec["rounds"]["llm"]["arms"]) | {"fw_pi05_libero_10_lda"}
    assert all(len(v) == 500 for v in arms.values())
    assert "fw_pi05_libero_10_uniform" not in arms
    assert prov["kb_libero_10_cp1_mean_pool_lda"]["full_hit"] == {"episodes": 500, "steps": 1000, "stale_rows_ignored": 0}
    assert prov["kb_libero_10_cp1_mean_pool_lda"]["run_ids"] == ["runpool"]
    assert prov["fw_pi05_libero_10_lda"]["attempts"] == {"1": 500} and prov["fw_pi05_libero_10_lda"]["launch_binding"] == "legacy"
    report = S.summarise("libero_10", arms)
    assert len(report["pairs"]) == 15


def test_manifest_launch_glob_resolves_per_run_records(tmp_path):
    runs, finish = make_manifest(tmp_path, use_glob=True)
    spec = finish()
    arms, prov = S.load_manifest_suite("libero_10", spec, expected=500)
    assert prov["kb_libero_10_cp1_max_pool_lda"]["launches"] == [str(runs["pool"].launch_path)]
    runs["pool"].launch_path.unlink()
    with pytest.raises(SystemExit, match="launch must name at least one file"):
        S.load_manifest_suite("libero_10", spec, expected=500)


def test_manifest_refuses_missing_arm_and_missing_episode(tmp_path):
    runs, finish = make_manifest(tmp_path)
    runs["pool"].rows = [r for r in runs["pool"].rows if r["yaml_id"] != "kb_libero_10_cp1_max_pool_lda"]
    spec = finish()
    with pytest.raises(SystemExit, match="no accepted episode"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    runs, finish = make_manifest(tmp_path / "b")
    runs["llm"].drop("kb_libero_10_cp1_llm_l0_prefix_mean_pool", 3, 7)
    spec = finish()
    with pytest.raises(SystemExit, match="499 accepted episodes"):
        S.load_manifest_suite("libero_10", spec, expected=500)


def test_manifest_refuses_foreign_arm_in_run_journal(tmp_path):
    runs, finish = make_manifest(tmp_path)
    runs["pool"].add("kb_libero_spatial_cp1_mean_pool_lda", 0, 0, success=True)
    spec = finish()
    with pytest.raises(SystemExit, match="outside the manifest"):
        S.load_manifest_suite("libero_10", spec, expected=500)


def test_manifest_refuses_non_eval_rows_and_uid_identity_drift(tmp_path):
    arm = "kb_libero_10_cp1_mean_pool_lda"
    runs, finish = make_manifest(tmp_path)
    runs["pool"].drop(arm, 0, 0)
    runs["pool"].add(arm, 0, 0, success=True, phase="warmup")
    spec = finish()
    with pytest.raises(SystemExit, match="warmup"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    runs, finish = make_manifest(tmp_path / "b")
    for r in runs["pool"].rows:
        if r["yaml_id"] == arm:
            r["task_uid"] = r["task_uid"].replace(":eval:", ":warmup:")
            r["phase"] = "warmup"
    spec = finish()
    with pytest.raises(SystemExit, match="warmup"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    runs, finish = make_manifest(tmp_path / "c")
    runs["pool"].drop(arm, 0, 0)
    runs["pool"].add(arm, 0, 0, success=True, row_arm="kb_libero_10_cp1_max_pool_lda")
    spec = finish()
    with pytest.raises(SystemExit, match="does not match its row"):
        S.load_manifest_suite("libero_10", spec, expected=500)


def test_manifest_refuses_suite_or_pool_mismatch_and_smoke(tmp_path):
    runs, finish = make_manifest(tmp_path)
    spec = finish()
    run = runs["llm"]
    run.launch["suite"] = "libero_spatial"
    run.write_launch()
    with pytest.raises(SystemExit, match="launch suite"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    run.launch["suite"] = "libero_10"
    run.launch["apool"]["rollup_sha256"] = "f" * 64
    run.write_launch()
    with pytest.raises(SystemExit, match="A-pool rollup"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    run.launch["apool"]["rollup_sha256"] = ROLLUP["libero_10"]
    run.launch["smoke"] = True
    run.write_launch()
    with pytest.raises(SystemExit, match="smoke"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    run.launch["smoke"] = False
    run.launch["arms"] = []
    run.write_launch()
    with pytest.raises(SystemExit, match="did not include arms"):
        S.load_manifest_suite("libero_10", spec, expected=500)


def test_manifest_launch_must_bind_every_run(tmp_path):
    arm = "kb_libero_10_cp1_llm_l0_prefix_mean_pool"
    # No launch at all: an empty list is not "nothing to check".
    runs, finish = make_manifest(tmp_path)
    spec = finish()
    spec["rounds"]["llm"]["launch"] = []
    with pytest.raises(SystemExit, match="launch must name at least one file"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    # A run source's record without run_id cannot bind anything.
    runs, finish = make_manifest(tmp_path / "b")
    spec = finish()
    runs["llm"].launch.pop("run_id")
    runs["llm"].write_launch()
    with pytest.raises(SystemExit, match="no run_id"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    # Accepted episodes from a run that has no checked launch record.
    runs, finish = make_manifest(tmp_path / "c")
    runs["llm"].drop(arm, 5, 5)
    runs["llm"].add(arm, 5, 5, success=True, run_id="runX")
    spec = finish()
    with pytest.raises(SystemExit, match="no checked launch record"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    # ...and it passes once that run's launch record is listed too.
    extra = Run(tmp_path / "c" / "llm2", "libero_10", [], run_id="runX", launch_arms=[arm])
    spec["rounds"]["llm"]["launch"] = [spec["rounds"]["llm"]["launch"], str(extra.launch_path)]
    arms, prov = S.load_manifest_suite("libero_10", spec, expected=500)
    assert prov[arm]["run_ids"] == ["runX", "runllm"]
    # A run source may not opt into the legacy contract.
    spec["rounds"]["llm"]["launch_binding"] = "legacy"
    with pytest.raises(SystemExit, match="legacy launch contract"):
        S.load_manifest_suite("libero_10", spec, expected=500)


def test_manifest_accepted_retry_needs_per_step_at_its_attempt(tmp_path):
    arm = "kb_libero_10_cp1_mean_pool_lda"
    runs, finish = make_manifest(tmp_path)
    runs["pool"].drop(arm, 1, 1)
    runs["pool"].add(arm, 1, 1, success=False, attempt=1, accepted=False)
    runs["pool"].add(arm, 1, 1, success=True, attempt=2, accepted=True)
    spec = finish()
    arms, prov = S.load_manifest_suite("libero_10", spec, expected=500)
    assert arms[arm][(1, 1)] is True
    assert prov[arm]["attempts"] == {"1": 499, "2": 1}
    assert prov[arm]["full_hit"]["stale_rows_ignored"] == 0  # fenced rows are not evidence at all
    runs, finish = make_manifest(tmp_path / "b")
    runs["pool"].drop(arm, 1, 1)
    runs["pool"].add(arm, 1, 1, success=True, attempt=2, accepted=True, step_attempt=1, step_accepted=True)
    spec = finish()
    with pytest.raises(SystemExit, match="no inference rows at their accepted"):
        S.load_manifest_suite("libero_10", spec, expected=500)


def test_manifest_per_step_must_come_from_the_accepted_run(tmp_path):
    arm = "kb_libero_10_cp1_max_pool_lda"
    # Journal from the new run, every per-step row from an older run at the same attempt.
    runs, finish = make_manifest(tmp_path)
    for r in runs["pool"].steps:
        if r["yaml_id"] == arm:
            r["run_id"] = "runOLD"
    spec = finish()
    with pytest.raises(SystemExit, match="only have rows from another run/attempt"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    # Rows the scheduler fenced are not evidence.
    runs, finish = make_manifest(tmp_path / "b")
    for r in runs["pool"].steps:
        if r["yaml_id"] == arm:
            r["accepted"] = False
    spec = finish()
    with pytest.raises(SystemExit, match="no inference rows"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    # Rows without the driver's run stamp cannot witness anything.
    runs, finish = make_manifest(tmp_path / "c")
    for r in runs["pool"].steps:
        r.pop("run_id", None)
    spec = finish()
    with pytest.raises(SystemExit, match="lacks attempt/run_id/step_idx"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    # Stale rows from another run are counted, never substituted.
    runs, finish = make_manifest(tmp_path / "d")
    runs["pool"].add(arm, 2, 2, success=True, run_id="runOLD")  # duplicate uid at same attempt, other run
    spec = finish()
    with pytest.raises(SystemExit, match="disagree"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    runs, finish = make_manifest(tmp_path / "e")
    runs["pool"].steps += [dict(r, run_id="runOLD") for r in runs["pool"].steps if r["task_uid"] == _uid(arm, 2, 2)]
    spec = finish()
    arms, prov = S.load_manifest_suite("libero_10", spec, expected=500)
    assert prov[arm]["full_hit"]["stale_rows_ignored"] == 2


def test_manifest_refuses_worker_error_and_non_full_hit(tmp_path):
    arm = "kb_libero_10_cp1_llm_l0_prefix_mean_pool"
    runs, finish = make_manifest(tmp_path)
    runs["llm"].drop(arm, 2, 2)
    runs["llm"].add(arm, 2, 2, success=False, error="RuntimeError: CUDA error")
    spec = finish()
    with pytest.raises(SystemExit, match="worker error"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    runs, finish = make_manifest(tmp_path / "b")
    runs["llm"].drop(arm, 2, 2)
    runs["llm"].add(arm, 2, 2, success=True, steps=("FULL_HIT", "MISS"))
    spec = finish()
    with pytest.raises(SystemExit, match="non-FULL_HIT"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    runs, finish = make_manifest(tmp_path / "c")
    runs["llm"].steps.append(dict(runs["llm"].steps[0]))  # same step twice in one file
    spec = finish()
    with pytest.raises(SystemExit, match="duplicate per-step row"):
        S.load_manifest_suite("libero_10", spec, expected=500)


def test_manifest_cross_file_duplicate_identical_ok_conflict_refused(tmp_path):
    arm = "kb_libero_10_cp1_max_pool_lda"
    runs, finish = make_manifest(tmp_path)
    spec = finish()
    pool_arms = spec["rounds"]["pool"]["arms"]
    # A resumed pool round in a second directory (same run id), one episode repeated identically.
    dup = Run(tmp_path / "pool2", "libero_10", [], launch_arms=pool_arms, run_id="runpool")
    dup.add(arm, 4, 4, success=(4 + 4) % 3 == 0)
    j2, p2 = dup.write()
    single = dict(spec["rounds"]["pool"])
    spec["rounds"]["pool"]["journal"] = [single["journal"], str(j2)]
    spec["rounds"]["pool"]["per_step"] = [single["per_step"], str(p2)]
    spec["rounds"]["pool"]["launch"] = [single["launch"], str(dup.launch_path)]
    arms, prov = S.load_manifest_suite("libero_10", spec, expected=500)
    assert len(arms[arm]) == 500 and prov[arm]["journal"] == spec["rounds"]["pool"]["journal"]
    # Same uid, different outcome in the second file: not resolvable by file order.
    dup.rows[0]["success"] = not dup.rows[0]["success"]
    dup.rows[0]["status"] = "done" if dup.rows[0]["success"] else "failed"
    dup.write()
    with pytest.raises(SystemExit, match="different"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    # Same outcome from a different run: a source-identity conflict, not a duplicate.
    dup.rows[0]["success"] = not dup.rows[0]["success"]
    dup.rows[0]["status"] = "done" if dup.rows[0]["success"] else "failed"
    dup.rows[0]["run_id"] = "runpool2"
    dup.write()
    with pytest.raises(SystemExit, match="different"):
        S.load_manifest_suite("libero_10", spec, expected=500)
    # Same uid/run/attempt but the second file holds a different step set (step 0 vs step 99).
    dup.rows[0]["run_id"] = "runpool"
    dup.steps = [dict(dup.steps[0], step_idx=99)]
    dup.write()
    with pytest.raises(SystemExit, match="different traces"):
        S.load_manifest_suite("libero_10", spec, expected=500)


def test_manifest_cli_is_exclusive(tmp_path, monkeypatch):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"suites": {}}))
    monkeypatch.setattr(sys, "argv", ["x", "--input-manifest", str(manifest), "--pi05-journal", "a=b",
                                      "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="exclusive"):
        S.main()


def test_manifest_cli_reports_pairs_as_exploratory(tmp_path, monkeypatch):
    runs, finish = make_manifest(tmp_path)
    spec = finish()
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"suites": {"libero_10": spec}, "episodes_per_arm": 500, "clip_deferred": False}))
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["x", "--input-manifest", str(manifest), "--out", str(out)])
    S.main()
    report = json.loads(out.read_text())
    assert "uncorrected" in report["manifest"]["pairs_note"]
    assert report["pi05_libero_10"]["arms"]["fw_pi05_libero_10_lda"]["n"] == 500


def test_scripts_run_by_path_without_pythonpath(tmp_path):
    """The lane flags and the emitter must work as plain script invocations (no PYTHONPATH)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for script in ("exp/weighted_sum/analysis/lcw_ablation_summary.py", "exp/weighted_sum/emit_keybuilder_lda.py"):
        proc = subprocess.run([sys.executable, str(REPO / script), "--help"], env=env, cwd=str(REPO),
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-800:]
    # The legacy lane entry end to end on a synthetic journal.
    run = Run(tmp_path / "j", "libero_spatial", ["a_arm", "b_arm"], legacy=True)
    j, _ = run.write()
    out = tmp_path / "o.json"
    proc = subprocess.run([sys.executable, str(REPO / "exp/weighted_sum/analysis/lcw_ablation_summary.py"),
                           "--pi05-journal", f"libero_spatial={j}", "--out", str(out)],
                          env=env, cwd=str(REPO), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert set(json.loads(out.read_text())["pi05_libero_spatial"]["arms"]) == {"a_arm", "b_arm"}


def test_driver_launch_records_carry_run_id_and_survive_relaunch(tmp_path):
    from exp.ablation_study.cache_size.run_size_eval import write_launch_records

    per_step = tmp_path / "per_step.jsonl"
    base = {"suite": "libero_10", "arms": ["x"], "trials_per_task": 50, "smoke": False, "apool": {"rollup_sha256": "r"}}
    first = write_launch_records(per_step, base, "run1")
    second = write_launch_records(per_step, base, "run2")
    assert first.name == "per_step.jsonl.launch.run1.json" and second.name == "per_step.jsonl.launch.run2.json"
    assert json.loads(first.read_text())["run_id"] == "run1"
    assert json.loads((tmp_path / "per_step.jsonl.launch.json").read_text())["run_id"] == "run2"
    assert "run_id" not in base


# ------------------------------------------------------------------
# Fit script checks
# ------------------------------------------------------------------
def _ss(method="zscore", squash="tanh", drop=None, sigma=0.01):
    fields = {f: {"method": method, "params": {"mu": 0.9, "sigma": sigma, "squash": squash}} for f in E.FIELDS}
    sim = {f: {"type": "l2" if f == "robot_state" else "cosine"} for f in E.FIELDS}
    if drop:
        fields.pop(drop)
    return {"type": "weighted_score_sum_knn", "field_similarity": sim,
            "score_normalization": {"type": "per_field", "fields": fields}}


def test_fit_template_checks():
    norm = F.check_template(_ss())
    assert set(norm) == set(E.FIELDS) and norm["robot_state"]["sim_type"] == "l2"
    for bad, msg in [(_ss(method="logit"), "zscore"), (_ss(drop="vision_1"), "lacks"), (_ss(sigma=0.0), "unusable")]:
        with pytest.raises(SystemExit, match=msg):
            F.check_template(bad)
    with pytest.raises(SystemExit, match="per_field"):
        F.check_template({"score_normalization": {"type": "percentile"}})


def test_fit_progress_and_degeneracy_checks():
    import numpy as np

    F.check_progress(np.array([0, 0, 0, 1, 1]), np.array([0, 1, 2, 0, 1]))
    with pytest.raises(SystemExit, match="not 0"):
        F.check_progress(np.array([0, 0, 0]), np.array([0, 2, 3]))
    assert F.safe_normalize([1.0, -1.0, 2.0]) == pytest.approx([1 / 3, 0.0, 2 / 3])
    assert F.safe_normalize([-1.0, -2.0, 0.0]) is None
    assert F.degenerate([-1.0, -2.0, 0.0], None)
    assert F.degenerate([float("nan"), 1.0, 1.0], [0.5, 0.5, 0.0])
    assert not F.degenerate([1.0, -1.0, 2.0], [1 / 3, 0.0, 2 / 3])


def test_fit_empty_population_is_reported_not_crashed():
    import numpy as np

    N = {f: np.zeros((2, 2)) for f in E.FIELDS}
    M = np.zeros((2, 2), dtype=bool)
    fl = F.fisher_and_lda(N, M, np.array([0.0, 1.0]), 0.05)
    assert fl["queries"] == 0 and fl["lda_raw"] is None
    pw = F.pairwise_logistic(N, M, np.array([0.0, 1.0]), 0.05, 10)
    assert pw["pairs"] == 0 and pw["w"] is None


# ------------------------------------------------------------------
# Launcher (stubbed tools; nothing real is started or killed)
# ------------------------------------------------------------------
LANE = REPO / "exp/weighted_sum/ops/fw_lane_pi05.sh"


def _stub_tools(bindir: Path, log: Path) -> None:
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "tmux").write_text(
        '#!/usr/bin/env bash\n'
        '# stub: "new -s NAME -d CMD" runs CMD (the last argument) in the background; everything else is a no-op\n'
        'if [ "$1" = new ]; then nohup bash -c "${!#}" >/dev/null 2>&1 & fi\nexit 0\n')
    (bindir / "pkill").write_text('#!/usr/bin/env bash\nexit 0\n')
    (bindir / "ss").write_text(
        '#!/usr/bin/env bash\nfor p in ${STUB_PORTS//,/ }; do echo "LISTEN 0 5 0.0.0.0:$p 0.0.0.0:*"; done\n')
    (bindir / "py").write_text(
        '#!/usr/bin/env bash\n'
        f'echo "$*" >> "{log}"\n'
        'if [ "$1" = scripts/serve_policy.py ]; then echo "Loaded 3 entries from stub"; sleep 1; fi\n'
        'exit 0\n')
    for f in bindir.iterdir():
        f.chmod(0o755)


def _lane(tmp: Path, suite: str, env_extra: dict, *, ports: str = "23280") -> tuple[int, str, list[str]]:
    bindir, log = tmp / "bin", tmp / "calls.log"
    _stub_tools(bindir, log)
    root = tmp / "root"
    (root / "scripts").mkdir(parents=True)
    cfg = tmp / "cfg" / suite
    cfg.mkdir(parents=True)
    (cfg / f"matrix_{suite}.yaml").write_text("suite: x\narms: []\n")
    (cfg / f"fw_pi05_{suite}_uniform.yaml").write_text("enabled: true\n")
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "HOME": str(tmp), "STUB_PORTS": ports,
           "FW_ROOT": str(root), "FW_PY": str(bindir / "py"), "FW_CFG_DIR": str(tmp / "cfg"),
           "FW_OUT": str(tmp / "out"), **env_extra}
    proc = subprocess.run(["bash", str(LANE), "127.0.0.1", "23290", ports, "2", suite],
                          env=env, capture_output=True, text=True, timeout=120)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc.returncode, proc.stdout + proc.stderr, calls


def test_launcher_defaults_unchanged(tmp_path):
    rc, out, calls = _lane(tmp_path, "libero_spatial", {})
    assert rc == 0, out
    serve = [c for c in calls if c.startswith("scripts/serve_policy.py")]
    drive = [c for c in calls if c.startswith("-m exp.ablation_study.cache_size.run_size_eval")]
    assert len(serve) == 1 and len(drive) == 1
    assert "--stage2-device meta --stage3-device meta" in serve[0]
    assert f"--cache-config {tmp_path}/cfg/libero_spatial/fw_pi05_libero_spatial_uniform.yaml" in serve[0]
    assert f"--arm-matrix {tmp_path}/cfg/libero_spatial/matrix_libero_spatial.yaml" in drive[0]
    assert "--eval-concurrency 2 " in drive[0] and "--role driver" in drive[0]
    logs = sorted((tmp_path / "out" / "libero_spatial").glob("*.log"))
    assert any(p.name.startswith("server_23280.") for p in logs) and any(p.name.startswith("driver.") for p in logs)
    assert "DRIVER_EXIT=0" in out


def test_launcher_forwards_stage2_matrix_boot_and_concurrency(tmp_path):
    kb = tmp_path / "kb" / "libero_10"
    kb.mkdir(parents=True)
    (kb / "matrix_libero_10_llm.yaml").write_text("suite: x\narms: []\n")
    (kb / "boot.yaml").write_text("enabled: true\n")
    rc, out, calls = _lane(tmp_path, "libero_10", {
        "FW_MATRIX": str(kb / "matrix_libero_10_llm.yaml"), "FW_BOOT": str(kb / "boot.yaml"),
        "FW_STAGE2_DEVICE": "cuda:0", "FW_EVAL_CONC": "4"})
    assert rc == 0, out
    serve = [c for c in calls if c.startswith("scripts/serve_policy.py")][0]
    drive = [c for c in calls if c.startswith("-m exp.ablation_study.cache_size.run_size_eval")][0]
    assert "--stage2-device cuda:0 --stage3-device meta" in serve
    assert f"--cache-config {kb}/boot.yaml" in serve
    assert f"--arm-matrix {kb}/matrix_libero_10_llm.yaml" in drive and "--eval-concurrency 4 " in drive


def test_launcher_runs_the_emitted_llm_round(world):
    """Plan section 3.4's LLM round on the emitter's own products: matrix + boot names must line up."""
    world.templates()
    world.fit_all()
    world.final()
    suite = "libero_10"
    matrix = world.cfg / suite / f"matrix_{suite}_llm.yaml"
    boot = world.yaml(suite, E.LLM_STEM)
    assert yaml.safe_load(matrix.read_text())["arms"][0]["arm"] == boot.stem == f"kb_{suite}_cp1_llm_l0_prefix_mean_pool"
    rc, out, calls = _lane(world.root / "lane", suite, {
        "FW_MATRIX": str(matrix), "FW_BOOT": str(boot), "FW_STAGE2_DEVICE": "cuda:0", "FW_EVAL_CONC": "1"})
    assert rc == 0, out
    serve = [c for c in calls if c.startswith("scripts/serve_policy.py")][0]
    drive = [c for c in calls if c.startswith("-m exp.ablation_study.cache_size.run_size_eval")][0]
    assert f"--cache-config {boot}" in serve and "--stage2-device cuda:0" in serve
    assert f"--arm-matrix {matrix}" in drive and "--eval-concurrency 1 " in drive


def test_launcher_refuses_relative_paths_and_bad_stage2_before_launching(tmp_path):
    rc, out, calls = _lane(tmp_path, "libero_10", {"FW_MATRIX": "relative/matrix.yaml"})
    assert rc == 2 and "absolute path" in out and calls == []
    rc, out, calls = _lane(tmp_path / "b", "libero_10", {"FW_STAGE2_DEVICE": "cuda:1"})
    assert rc == 2 and "FW_STAGE2_DEVICE" in out and calls == []
    rc, out, calls = _lane(tmp_path / "c", "libero_10", {"FW_BOOT": str(tmp_path / "c" / "missing.yaml")})
    assert rc == 2 and "missing" in out and calls == []
