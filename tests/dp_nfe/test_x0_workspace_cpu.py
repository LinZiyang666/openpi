"""CPU tests of ``FixedStepWorkspace`` control flow with the DP stand-ins of ``tests/dp_nfe/dp_stubs.py`` (installed only
when the real ``diffusion_policy`` is absent; inside the DP env the same tests run against the real base classes with
the stub policy/dataset). Covers G2 R1-B2 / R1-B6: exact budget for both heads, resume after an interruption reproducing
the uninterrupted run (parameters, EMA, losses, LR), refusal of a payload with a conflicting or missing identity, the
frozen training-pool normaliser shared by U/M cells with hash verification, and the trainer's manifest non-overwrite."""

import copy
import json
import pathlib

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tests.dp_nfe import dp_stubs

dp_stubs.install()

from exp.dp_nfe import x0_normalizer  # noqa: E402
from exp.dp_nfe.x0_identity import identity_diff  # noqa: E402
from exp.dp_nfe.x0_workspace import FixedStepWorkspace  # noqa: E402


def _episodes(path: pathlib.Path, n: int, T: int, seed: int, scale=1.0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, T, 3)) * scale; act = rng.normal(size=(n, T, 2)) * scale * 0.5
    np.savez(path, obs=obs, action=act)
    return path


def _cfg(train_npz, heldout_npz, out, cell_id="cellU", head="epsilon", budget=6, save_every=100, identity=None,
         normalizer_path=None, normalizer_sha=None, seed=42, num_workers=0):
    ident = {"task_name": "toy", "modality": "lowdim", "variant": "full", "budget_id": "Btest", "subset_sha256": "s0",
             "resolved_config_sha256": "c0", "code_sha256": "k0", "deps": "{}"}
    ident.update(identity or {})
    c = {"_target_": "exp.dp_nfe.x0_workspace.FixedStepWorkspace",
         "training": {"seed": seed, "device": "cpu", "resume": True, "lr_scheduler": "constant", "use_ema": True},
         "policy": {"_target_": "tests.dp_nfe.dp_stubs.TinyPolicy",
                    "noise_scheduler": {"_target_": "tests.dp_nfe.dp_stubs.TinyScheduler", "prediction_type": head}},
         "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-3},
         "ema": {"_target_": "tests.dp_nfe.dp_stubs.EMAModel"},
         "task": {"name": "toy", "dataset": {"_target_": "tests.dp_nfe.dp_stubs.ArrayDataset", "path": str(train_npz), "horizon": 4}},
         "x0": {"cell_id": cell_id, "budget_steps": budget, "batch_size": 8, "window_seed": 7, "val_every": 2, "save_every": save_every,
                "val_batch_size": 8, "num_workers": num_workers, "identity": ident, "normalizer_path": normalizer_path, "normalizer_sha256": normalizer_sha,
                "heldout_dataset": {"_target_": "tests.dp_nfe.dp_stubs.ArrayDataset", "path": str(heldout_npz), "horizon": 4}}}
    return OmegaConf.create(c)


def _params(ws):
    return {k: v.detach().clone() for k, v in ws.ema_model.state_dict().items()}, {k: v.detach().clone() for k, v in ws.model.state_dict().items()}


@pytest.mark.parametrize("head", ["epsilon", "sample"])
def test_exact_budget_both_heads_and_final_artifacts(tmp_path, head):
    tr = _episodes(tmp_path / "tr.npz", 4, 12, 0); ho = _episodes(tmp_path / "ho.npz", 2, 12, 1)
    out = tmp_path / f"run_{head}"
    ws = FixedStepWorkspace(_cfg(tr, ho, out, head=head, budget=5), output_dir=str(out)); final = ws.run()
    assert ws.global_step == 5 and ws.ema.optimization_step == 5 and ws.finished and ws.samples_seen == 40
    done = json.loads((out / "checkpoints" / "final.done").read_text())
    assert done["global_step"] == 5 and done["finished"] and len(done["checkpoint_sha256"]) == 64
    assert pathlib.Path(final).is_file() and (out / "checkpoints" / "latest.ckpt").is_file()
    ident = json.loads((out / "identity.json").read_text())
    assert ident["head"] == head and ident["normalizer_source"] == "fitted" and ident["heldout_windows"] > 0
    log = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3, 4, 5] and all(np.isfinite(r["loss"]) for r in log)
    assert "val_mse_ema" in log[1] and "val_mse_ema" in log[-1]


def test_resume_after_interruption_reproduces_uninterrupted_run(tmp_path, monkeypatch):
    tr = _episodes(tmp_path / "tr.npz", 4, 12, 0); ho = _episodes(tmp_path / "ho.npz", 2, 12, 1)
    # A: uninterrupted, 6 updates
    outA = tmp_path / "A"
    wsA = FixedStepWorkspace(_cfg(tr, ho, outA, budget=6), output_dir=str(outA)); wsA.run()
    emaA, modA = _params(wsA)
    logA = [json.loads(l) for l in (outA / "train_log.jsonl").read_text().splitlines()]
    # B: same cell, latest.ckpt every 3 updates, killed right after the save at step 3
    outB = tmp_path / "B"
    orig = FixedStepWorkspace.save_atomic
    def killing_save(self, tag):
        p = orig(self, tag)
        if tag == "latest" and self.global_step == 3:
            raise KeyboardInterrupt("simulated kill after the step-3 save")
        return p
    monkeypatch.setattr(FixedStepWorkspace, "save_atomic", killing_save)
    with pytest.raises(KeyboardInterrupt):
        FixedStepWorkspace(_cfg(tr, ho, outB, budget=6, save_every=3), output_dir=str(outB)).run()
    monkeypatch.setattr(FixedStepWorkspace, "save_atomic", orig)
    assert (outB / "checkpoints" / "latest.ckpt").is_file() and not (outB / "checkpoints" / "final.ckpt").exists()
    # B': a fresh process resumes from latest and finishes updates 4..6
    wsB = FixedStepWorkspace(_cfg(tr, ho, outB, budget=6, save_every=3), output_dir=str(outB)); wsB.run()
    assert wsB.global_step == 6 and wsB.ema.optimization_step == 6
    emaB, modB = _params(wsB)
    for k in modA:
        assert torch.allclose(modA[k], modB[k], atol=0, rtol=0), k
        assert torch.allclose(emaA[k], emaB[k], atol=0, rtol=0), k
    logB = [json.loads(l) for l in (outB / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in logB] == [1, 2, 3, 4, 5, 6]
    for a, b in zip(logA, logB):
        assert a["loss"] == b["loss"] and a["lr"] == b["lr"] and a.get("val_mse_ema") == b.get("val_mse_ema")
    assert wsA.lr_scheduler.get_last_lr() == wsB.lr_scheduler.get_last_lr()


def test_worker_dataloader_fetches_the_same_batches(tmp_path):
    """num_workers=2 (worker processes assembling the fixed batches) trains bit-identically to the in-process loop."""
    tr = _episodes(tmp_path / "tr.npz", 4, 12, 0); ho = _episodes(tmp_path / "ho.npz", 2, 12, 1)
    ws0 = FixedStepWorkspace(_cfg(tr, ho, tmp_path / "w0", budget=5, num_workers=0), output_dir=str(tmp_path / "w0")); ws0.run()
    ws2 = FixedStepWorkspace(_cfg(tr, ho, tmp_path / "w2", budget=5, num_workers=2), output_dir=str(tmp_path / "w2")); ws2.run()
    for (k, a), (_, b) in zip(ws0.model.state_dict().items(), ws2.model.state_dict().items()):
        assert torch.equal(a, b), k
    l0 = [json.loads(l)["loss"] for l in (tmp_path / "w0" / "train_log.jsonl").read_text().splitlines()]
    l2 = [json.loads(l)["loss"] for l in (tmp_path / "w2" / "train_log.jsonl").read_text().splitlines()]
    assert l0 == l2 and len(l0) == 5


def test_conflicting_or_missing_identity_is_refused(tmp_path):
    tr = _episodes(tmp_path / "tr.npz", 4, 12, 0); ho = _episodes(tmp_path / "ho.npz", 2, 12, 1)
    out = tmp_path / "run"
    FixedStepWorkspace(_cfg(tr, ho, out, budget=4, save_every=2), output_dir=str(out)).run()
    latest = out / "checkpoints" / "latest.ckpt"
    # a different subset hash under the same cell name
    ws = FixedStepWorkspace(_cfg(tr, ho, out, budget=4, identity={"subset_sha256": "OTHER"}), output_dir=str(out))
    with pytest.raises(ValueError, match="subset_sha256"):
        ws.load_checkpoint(path=str(latest))
    # a different budget / code hash / normalizer hash
    for ident in ({"code_sha256": "k1"}, {"deps": '{"torch": "x"}'}):
        with pytest.raises(ValueError, match="identity mismatch"):
            FixedStepWorkspace(_cfg(tr, ho, out, budget=4, identity=ident), output_dir=str(out)).load_checkpoint(path=str(latest))
    with pytest.raises(ValueError, match="identity mismatch"):
        FixedStepWorkspace(_cfg(tr, ho, out, budget=5), output_dir=str(out)).load_checkpoint(path=str(latest))
    # a payload without identity
    import dill
    payload = torch.load(latest.open("rb"), pickle_module=dill)
    del payload["pickles"]["identity"]
    with pytest.raises(ValueError, match="no identity"):
        FixedStepWorkspace(_cfg(tr, ho, out, budget=4), output_dir=str(out)).load_payload(payload)
    # identity.json of another cell in the output dir is refused before training
    out2 = tmp_path / "run2"; out2.mkdir()
    (out2 / "identity.json").write_text(json.dumps({"cell_id": "someone_else"}))
    with pytest.raises(ValueError, match="different cell"):
        FixedStepWorkspace(_cfg(tr, ho, out2, budget=2), output_dir=str(out2)).run()


def test_frozen_normalizer_is_shared_by_um_cells_and_hash_verified(tmp_path):
    pool = _episodes(tmp_path / "pool.npz", 8, 12, 0, scale=3.0)
    U = _episodes(tmp_path / "U.npz", 3, 12, 5, scale=1.0); M = _episodes(tmp_path / "M.npz", 3, 12, 6, scale=0.2)
    ho = _episodes(tmp_path / "ho.npz", 2, 12, 1)
    ds = dp_stubs.ArrayDataset(str(pool))
    side = x0_normalizer.save_normalizer(ds.get_normalizer(), tmp_path / "toy_lowdim_normalizer.pt", {"src": "pool"})
    frozen = torch.load(str(tmp_path / "toy_lowdim_normalizer.pt"))
    assert x0_normalizer.normalizer_sha256(frozen) == side["sha256"] == x0_normalizer.read_sidecar(tmp_path / "toy_lowdim_normalizer.pt")["sha256"]
    states = {}
    for variant, data in (("U", U), ("M", M)):
        out = tmp_path / f"run_{variant}"
        ws = FixedStepWorkspace(_cfg(data, ho, out, cell_id=f"cell{variant}", budget=3, identity={"variant": variant, "subset_sha256": variant},
                                     normalizer_path=str(tmp_path / "toy_lowdim_normalizer.pt"), normalizer_sha=side["sha256"]), output_dir=str(out))
        ws.run()
        states[variant] = {k: v.clone() for k, v in ws.model.normalizer.state_dict().items()}
        ident = json.loads((out / "identity.json").read_text())
        assert ident["normalizer_source"] == "frozen" and ident["normalizer_sha256_loaded"] == side["sha256"]
        # the checkpoint carries the same normaliser inside the policy state_dict
        import dill
        payload = torch.load((out / "checkpoints" / "final.ckpt").open("rb"), pickle_module=dill)
        sd = {k[len("normalizer."):]: v for k, v in payload["state_dicts"]["model"].items() if k.startswith("normalizer.")}
        assert x0_normalizer.normalizer_sha256(sd) == side["sha256"]
    assert set(states["U"]) == set(states["M"]) == set(frozen)
    for k in frozen:
        assert torch.equal(states["U"][k], frozen[k]) and torch.equal(states["M"][k], frozen[k])
    # the subsets' own statistics differ from the pool's -> a per-cell fit would not have matched
    own = dp_stubs.ArrayDataset(str(U)).get_normalizer().state_dict()
    assert x0_normalizer.normalizer_sha256(own) != side["sha256"]
    # wrong expected hash / tampered file -> refused; U/M without a frozen file -> refused at construction
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="hash mismatch"):
        FixedStepWorkspace(_cfg(U, ho, out, budget=1, identity={"variant": "U"}, normalizer_path=str(tmp_path / "toy_lowdim_normalizer.pt"),
                                normalizer_sha="0" * 64), output_dir=str(out)).run()
    with pytest.raises(ValueError, match="frozen training-pool normalizer"):
        FixedStepWorkspace(_cfg(U, ho, out, budget=1, identity={"variant": "M"}), output_dir=str(out))


def test_write_manifest_never_overwrites_a_different_identity(tmp_path):
    from exp.dp_nfe import train_x0
    tr = _episodes(tmp_path / "tr.npz", 2, 8, 0); ho = _episodes(tmp_path / "ho.npz", 1, 8, 1)
    cfg = _cfg(tr, ho, tmp_path, budget=2)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    cell = {"cell_id": "cellU", "head": "epsilon", "task": "toy", "train_seed": 42}
    out = tmp_path / "m"; out.mkdir()
    m1 = train_x0.write_manifest(out, cell, cfg, resolved)
    raw = (out / "manifest.json").read_bytes()
    assert m1["identity"]["subset_sha256"] == "s0" and (out / "resolved_config.yaml").is_file()
    # same identity -> kept byte-for-byte; different identity -> refused, file untouched
    train_x0.write_manifest(out, cell, cfg, resolved)
    assert (out / "manifest.json").read_bytes() == raw
    cfg2 = copy.deepcopy(cfg); cfg2.x0.identity.subset_sha256 = "s1"
    with pytest.raises(SystemExit):
        train_x0.write_manifest(out, cell, cfg2, OmegaConf.to_container(cfg2, resolve=True))
    assert (out / "manifest.json").read_bytes() == raw
    assert identity_diff({"a": 1}, {"a": "1"}) == [] and identity_diff({"a": 1}, {}) == ["a: 1 != '<missing>'"]


def test_finished_resume_preserves_final_bytes_and_log(tmp_path):
    tr = _episodes(tmp_path / "tr.npz", 2, 8, 0); ho = _episodes(tmp_path / "ho.npz", 1, 8, 1)
    cfg = _cfg(tr, ho, tmp_path, budget=2)
    ws = FixedStepWorkspace(cfg, output_dir=str(tmp_path)); final = ws.run()
    before = final.read_bytes(); log = (tmp_path / "train_log.jsonl").read_bytes()
    resumed = FixedStepWorkspace(cfg, output_dir=str(tmp_path)); resumed.run()
    assert final.read_bytes() == before and (tmp_path / "train_log.jsonl").read_bytes() == log
    assert resumed.finished and resumed.global_step == 2
