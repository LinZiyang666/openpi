import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from exp.offline_search.debug.catalog import build_catalog, catalog_key, LIB_FIELDS, main
from exp.offline_search.debug.fixtures import make_synthetic_arm
from exp.offline_search.debug import reader
from exp.offline_search.rounds.r07.stages.stages import StageTable, content_fingerprint


def store_library(root, library="current", n=6):
    directory = root / "library/pi05_l10" / library
    directory.mkdir(parents=True)
    values = dict(key_v0=np.ones((n, 4), np.float32), key_v1=np.zeros((n, 4), np.float32),
                  rs=np.arange(n * 32, dtype=np.float32).reshape(n, 32),
                  action=np.zeros((n, 10, 32), np.float32), task_id=np.zeros(n, np.int32),
                  episode=np.arange(n) // 3, step=np.arange(n) % 3, ep_len=np.full(n, 3, np.int32),
                  progress=(np.arange(n) % 3) / 2., success=np.ones(n, bool),
                  prev=np.where(np.arange(n) % 3 == 0, -1, np.arange(n) - 1),
                  next=np.where(np.arange(n) % 3 == 2, -1, np.arange(n) + 1))
    for name, value in values.items():
        np.save(directory / (name + ".npy"), value, allow_pickle=False)
    return values


def frozen_stage(values):
    table = StageTable()
    table.manifest = dict(exec_steps=5, H=10, act_valid_dims=7, rs_valid_dims=8, gripper_dim=6)
    table.fingerprint = content_fingerprint(values, table.manifest)
    table.retrieval_fingerprint = "frozen-synthetic-A"
    for key in ("task_id", "episode", "step", "success"):
        setattr(table, key, values[key].copy())
    n = len(table.task_id)
    table.mode = (np.arange(n) % 2).astype(np.int8)
    table.event_near = np.arange(n) % 3 == 1
    table.rows_to_event = np.full(n, 1, np.int32)
    table.stage_run = np.zeros(n, np.int32)
    return table


def test_catalog_uses_frozen_embedded_stage_and_hashes_store_inputs(tmp_path):
    store, run, fits = tmp_path / "store", tmp_path / "run", tmp_path / "fits"
    values = store_library(store)
    fits.mkdir()
    table = frozen_stage(values)
    artifact = fits / "r7_pi05_l10_50_SF1.pkl"
    with artifact.open("wb") as f:
        pickle.dump(dict(method=SimpleNamespace(base=SimpleNamespace(follow_table=table))), f)
    report = build_catalog(run, "pi05_l10_current", store, fits)
    output = run / "catalog/pi05_l10_current"
    frame = pd.read_parquet(output / "rows.parquet")
    assert frame.row.tolist() == list(range(6))
    for field in ("mode", "event_near", "rows_to_event", "stage_run"):
        assert np.array_equal(frame[field], getattr(table, field))
    assert report["stages"]["status"] == "available"
    assert report["stages"]["fingerprint"] == table.fingerprint
    assert report == json.loads((output / "rows.json").read_text())
    assert report["sha256"] == hashlib.sha256((output / "rows.parquet").read_bytes()).hexdigest()
    assert set(report["store_arrays"]) == set(LIB_FIELDS)
    for spec in report["store_arrays"].values():
        assert spec["sha256"] == hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()
    assert set(frame.lib_sha) == {report["lib_sha"]}


def test_current_and_big_catalogs_are_separate_cli_outputs(tmp_path, monkeypatch, capsys):
    store, run, fits = tmp_path / "store", tmp_path / "run", tmp_path / "no_fits"
    store_library(store)
    store_library(store, "bpool_cs", n=9)
    monkeypatch.setattr("sys.argv", ["catalog", "--run-root", str(run), "--store-root", str(store),
                                     "--r7-fits", str(fits), "--cells", "pi05_l10_current", "pi05_l10_bpool_cs"])
    main()
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["rows"] for r in reports] == [6, 9]
    assert all(r["stages"]["status"] == "unavailable" for r in reports)
    assert reports[0]["lib_sha"] != reports[1]["lib_sha"]
    assert "mode" not in pd.read_parquet(run / "catalog/pi05_l10_current/rows.parquet")
    assert catalog_key("pi05", "libero_10", "bpool_cs") == "pi05_l10_bpool_cs"


def test_wrong_library_stage_is_rejected_before_publication(tmp_path):
    store, run = tmp_path / "store", tmp_path / "run"
    values = store_library(store)
    store_library(store, "bpool_cs", n=9)
    stage = tmp_path / "stage.pkl"
    frozen_stage(values).save(stage)
    with pytest.raises(ValueError, match="different library"):
        build_catalog(run, "pi05_l10_bpool_cs", store, tmp_path, stage)
    assert not (run / "catalog").exists()
    report = build_catalog(run, "pi05_l10_current", store, tmp_path, stage)
    assert report["stages"]["status"] == "available"


def test_reader_requires_library_key_and_never_uses_serving_cell(tmp_path):
    arm = reader.open_arm(make_synthetic_arm(tmp_path / "run"), "synthetic")
    current = arm.run_root / arm.manifest.pop("catalog")
    legacy = current.with_name("pi05_synthetic_cache")
    current.rename(legacy)
    arm.manifest["arm_spec"] = dict(cell=legacy.name)
    assert arm.catalog() is None
    current.mkdir()
    frame = pd.read_parquet(legacy / "rows.parquet")
    frame.to_parquet(current / "rows.parquet", index=False)
    big = current.with_name("pi05_synthetic_bpool_cs")
    big.mkdir()
    frame.iloc[:5].to_parquet(big / "rows.parquet", index=False)
    assert len(arm.catalog()) == 16
    arm.manifest["lib"] = "bpool_cs"
    assert len(arm.catalog()) == 5
    arm.manifest["catalog"] = str(current)
    with pytest.raises(ValueError, match="model/suite/library"):
        arm.catalog()
