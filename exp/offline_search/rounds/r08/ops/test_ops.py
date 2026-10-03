"""Selection, frozen-source rejection and prefit artifact contracts."""
import json
from pathlib import Path
import tempfile
import unittest

from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sha
from .emit_arms import CELLS, COUNTS, make_specs, smoke_specs, verify_source, writable_output
from .prefit import output_root, publish
from .gpu_commands import commands


def fake_lock():
    cells = {}
    for cell in CELLS:
        model, suite, size = cell.split("_")
        base = dict(method="base:BlindAWM", artifact="/frozen/" + cell + ".pkl", sha256="abc", dependencies={},
                    kwargs=dict(lib="current" if size == "50" else "big", kref=5,
                                serving="anchor_tail", budget=1, gates="budget_only"))
        cells[cell] = dict(A=base, CU=dict(base, method="calls:CallController", kwargs=dict(rho=.3 if size == "50" else .18, tilt=False, random_seed=26092903)),
                          CT=dict(base, method="calls:CallController", kwargs=dict(rho=.3 if size == "50" else .18, tilt=True, random_seed=26092903)),
                          SF1=dict(base, method="follow:StageFollow", kwargs=dict(base["kwargs"], extend_blocks=1)))
        if model == "pi05":
            cells[cell]["SW"] = dict(base, method="wrist:StageWrist", kwargs=dict(enabled=True, base_kwargs=base["kwargs"],
                base_fit=base["artifact"], wrist_fit="/frozen/wrist.pkl", stage_fit="/frozen/stage.pkl"))
    return dict(cells=cells)


class OpsTest(unittest.TestCase):
    def test_exact_70_cells_priorities_and_oracle_flags(self):
        rows = make_specs(fake_lock())
        self.assertEqual({v: sum(r["r8"]["variant"] == v for r in rows) for v in COUNTS}, COUNTS)
        self.assertEqual({p: sum(r["r8"]["priority"] == p for r in rows) for p in ("P0", "P1", "P2")},
                         dict(P0=32, P1=34, P2=4))
        self.assertEqual(rows[0]["model"], "groot")
        self.assertEqual(rows[0]["suite"], "l10")
        self.assertTrue(all(r["client_overrides"]["replan_steps"] == 5 for r in rows))
        self.assertTrue(all(r["r8"]["env_seed"] == 7 and r["r8"]["debug_required"] for r in rows))
        self.assertEqual(sum("--os-oracle" in r["plugin_args"] for r in rows), 6)
        self.assertEqual(sum(r["r8"]["diagnostic_oracle"] for r in rows), 6)
        self.assertEqual(len({r["name"] for r in rows}), 70)
        self.assertTrue(all(r["full_model"] for r in rows if r["r8"]["variant"] in ("P10", "CU", "CT", "IP", "O5a", "O5b")))
        self.assertTrue(all("--os-policy-tail" in r["plugin_args"] for r in rows if r["r8"]["variant"] == "P10"))
        self.assertTrue(all(r["r8"]["library_size"] is None for r in rows if r["r8"]["variant"] == "P10"))
        self.assertTrue(all(r["kwargs"]["random_seed"] != 26092903 for r in rows if r["r8"]["variant"] in ("CU", "CT")))

    def test_immutable_input_and_repeatable_emission(self):
        lock = fake_lock()
        before = json.dumps(lock, sort_keys=True)
        a = make_specs(lock)
        self.assertEqual(json.dumps(lock, sort_keys=True), before)
        self.assertEqual(a, make_specs(lock))

    def test_hash_tampering_detected(self):
        root = Path("/tmp/r8_S5/tests")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            path, dep = Path(tmp) / "fit.pkl", Path(tmp) / "calibration.json"
            path.write_bytes(b"frozen-fit")
            dep.write_bytes(b"frozen-calibration")
            source = dict(artifact=str(path), sha256=sha(path), dependencies={str(dep): sha(dep)})
            verify_source(source)
            dep.write_bytes(b"different")
            with self.assertRaises(ValueError):
                verify_source(source)
            path.write_bytes(b"different")
            with self.assertRaises(ValueError):
                verify_source(source, dependencies=False)

    def test_output_bounds_and_atomic_pickle_format(self):
        for path in ("/home/weiland/trace_runs/new", "/tmp/r8_S5/../escape", "/tmp/r8_S6/fits"):
            with self.assertRaises(ValueError):
                output_root(path)
        with self.assertRaises(ValueError):
            writable_output("/home/weiland/trace_runs/new")
        path = Path("/tmp/r8_S5/tests/published.pkl")
        publish(path, dict(spec="test:Method", kwargs={}, cell="pi05_l10_cache", method="fixture"))
        self.assertTrue(path.is_file())
        self.assertFalse(path.with_suffix(".pkl.part").exists())

    def test_smoke_manifest_hash_role_and_fit_identity(self):
        root = Path("/tmp/r8_S5/tests")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(dict(role="NONTEST_BVAL", selected=[dict(task=0, init=99)])))
            lock = fake_lock()
            lock["manifests"] = {"{}_{}_bval20".format(m, s): dict(path=str(path), sha256=sha(path))
                                 for m in ("pi05", "groot") for s in ("l10", "spatial")}
            rows = make_specs(lock)
            smoke = smoke_specs(rows, lock)
            for a, b in zip(rows, smoke):
                self.assertEqual(b["name"], a["name"] + "_smoke")
                self.assertEqual(a["kwargs"], b["kwargs"])
                self.assertEqual(a["plugin_args"], b["plugin_args"])
                self.assertEqual(b["manifest"], str(path))
            path.write_text(json.dumps(dict(role="OFFICIAL_TEST", selected=[])))
            with self.assertRaises(ValueError):
                smoke_specs(rows, lock)
            for source in lock["manifests"].values():
                source["sha256"] = sha(path)
            with self.assertRaises(ValueError):
                smoke_specs(rows, lock)

    def test_gpu_command_generator_never_uses_reserved_ports(self):
        arm = dict(model="pi05", method="method:Policy", kwargs={}, cell="pi05_l10_cache",
                   plugin_args=[], yaml="/tmp/r8_S5/config.yaml", suite_short="l10")
        off = commands(arm, 23240, "/tmp/r8_S5/gpu/off", False)
        on = commands(arm, 23240, "/tmp/r8_S5/gpu/on", True)
        self.assertNotIn("--os-debug-dir", off["server"])
        self.assertIn("--os-debug-dir", on["server"])
        self.assertIn("policy:checkpoint", on["server"])
        groot = commands(dict(arm, model="groot", cell="groot_l10_cache"), 23241, "/tmp/r8_S5/gpu/groot", True)
        self.assertIn("/data/ckpt/n15_libero_10", groot["server"])
        self.assertNotIn("--stage1-only", groot["server"])
        for port in (23100, 23150, 23199, 0):
            with self.assertRaises(ValueError):
                commands(arm, port, "/tmp/r8_S5/gpu", False)


if __name__ == "__main__":
    unittest.main()
