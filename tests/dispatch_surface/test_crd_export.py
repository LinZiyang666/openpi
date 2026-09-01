"""Export -> emit -> config-load chain for H-CRD artifacts on the synthetic Rev 1 world (G2 R1 B6)."""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from exp.dispatch_surface import export_crd_artifacts as crd_export
from exp.dispatch_surface import sgrid_sweep
from exp.dispatch_surface.fit_surface import load_table
from openpi.cache import config as cfgmod
from openpi.cache.components import crd_judge as crd
from openpi.cache.components.surface_judge import load_surface_artifact, surface_verdict
from tests.dispatch_surface.test_rev2_phase0 import TEMPLATE, build_package, build_world


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("crd")
    w = build_world(tmp)
    manifest_path, _ = build_package(w, tmp)
    return types.SimpleNamespace(tmp=tmp, world=w, manifest_path=manifest_path)


def _export(world, out_dir, **kw):
    args = types.SimpleNamespace(rev1_package_manifest=str(world.manifest_path), source_role="artifact.dsp_sv",
                                 table=str(world.world.table), quantiles="0.85", gammas="1.0", budget_mults="2,inf",
                                 j_bad="3,inf", l_max="6,none", min_recovery_misses=2, out_dir=str(out_dir))
    for k, v in kw.items():
        setattr(args, k, v)
    return crd_export.export(args)


@pytest.fixture(scope="module")
def exported(world):
    return _export(world, world.tmp / "crd_sv")


def test_export_produces_the_parameter_cross_product(exported):
    names = sorted(exported["artifacts"])
    assert len(names) == 8                                     # 1 q x 1 gamma x 2 mults x 2 j x 2 L
    assert "crd_q85_g1_m2_j3_L6" in names and "crd_q85_g1_minf_jinf_Lnone" in names


def test_export_rejects_invalid_recovery_dwell_before_writing_artifacts(world):
    out_dir = world.tmp / "bad-recovery-dwell"
    with pytest.raises(SystemExit, match="min-recovery-misses"):
        _export(world, out_dir, min_recovery_misses=-1)
    assert out_dir.is_dir() and not any(out_dir.iterdir())


def test_artifacts_carry_consistent_grids_and_task_scales(world, exported):
    table = load_table(str(world.world.table))
    tasks = sorted(set(int(t) for t in table.task))
    for name, art in exported["artifacts"].items():
        with np.load(art["path"], allow_pickle=False) as data:
            u, d = data["q_hat"], data["q_hat_central"]
            meta = json.loads(bytes(data["meta_json"]).decode())
        c = meta["crd"]
        assert meta["judge_variant"] == crd.JUDGE_VARIANT
        assert np.all(d <= u + 1e-12) and np.all(np.isfinite(d)) and np.all(d >= 0)
        assert c["upper_grid_sha256"] == crd.grid_sha256(u) and c["central_grid_sha256"] == crd.grid_sha256(d)
        assert sorted(int(k) for k in c["task_scale"]) == tasks
        assert all(np.isfinite(v) and v > 0 for v in c["task_scale"].values())
        assert c["min_recovery_misses"] == 2
        judge = crd.CumulativeRiskJudge(art["path"])           # loads under the fail-fast validator
        assert judge.delta == pytest.approx(art["delta"])


def test_mutated_artifact_is_refused(exported, tmp_path):
    art = next(iter(exported["artifacts"].values()))
    with np.load(art["path"], allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["q_hat_central"] = arrays["q_hat_central"].copy()
    arrays["q_hat_central"][1, 0, 0] = -1.0
    bad = tmp_path / "bad.npz"
    np.savez(bad, **arrays)
    with pytest.raises(ValueError):
        crd.CumulativeRiskJudge(str(bad))


def test_degenerate_parameters_reproduce_the_static_surface(exported):
    art = exported["artifacts"]["crd_q85_g1_minf_jinf_Lnone"]
    judge = crd.CumulativeRiskJudge(art["path"])
    judge.gamma = 0.0                                          # beta=inf, j=inf, L=none already
    surf = load_surface_artifact(art["path"])
    edges = surf.v_bin_edges
    for b in range(len(edges) - 1):
        v = edges[b] + 1e-6 if np.isfinite(edges[b]) else edges[b + 1] - 1e-6
        for s in np.linspace(0.0, 1.0, 401):
            static = surface_verdict(float(s), float(v), edges, surf.s_min_full, surf.s_min_warm,
                                     uses_disagreement=surf.uses_disagreement)
            u_f = judge._cell(judge._u, 1, float(s), b)
            u_w = judge._cell(judge._u, 0, float(s), b)
            mine = "full" if u_f <= judge.delta else ("warm" if u_w <= judge.delta else "miss")
            assert mine == static, (s, b)


def test_emit_and_config_load_route_to_the_stateful_judge(world, exported):
    out_dir = world.tmp / "crd_cfg"
    args = types.SimpleNamespace(rev1_package_manifest=str(world.manifest_path),
                                 export_records=str(world.tmp / "crd_sv" / "export_record.json"),
                                 template=str(TEMPLATE), library_pkl=str(world.world.lib), out_dir=str(out_dir))
    sgrid_sweep.emit(args)
    matrix = json.loads((out_dir / "arm_matrix_sgrid.json").read_text())
    assert len(matrix["arms"]) == 8 and matrix["protocol"] == sgrid_sweep.PROTOCOL_SGRID
    yaml_path = matrix["arms"]["dsp_sv_crd_q85_g1_m2_j3_L6"]
    cfg = cfgmod.load_cache_config(yaml_path)
    cp1 = cfg.checkpoints["cp1"]
    assert cp1.gate.type == "always_search" and cp1.judge.export_factor_outputs is True
    judge = cfgmod._build_judge(cp1.judge)
    assert isinstance(judge, crd.CumulativeRiskJudge) and judge.j_bad == 3 and judge.l_max == 6
    plain = cfgmod._build_judge(cfgmod.JudgeConfig(type="dispatch_surface",
                                                    surface_artifact_path=str(world.world.artifacts["dsp_sv"])))
    assert not isinstance(plain, crd.CumulativeRiskJudge)
