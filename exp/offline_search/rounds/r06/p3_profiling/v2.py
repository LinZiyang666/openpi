"""Version 2 of the same opt-in Profile mode. V1 import/fit/serving is untouched.

Use this module's Profile with enabled=true. enabled=false delegates exactly.
No privileged client information enters the method or assignment features.
"""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .method import Profile as V1, awm_of
from .design import Design, validate
from exp.offline_search.closed_loop.blind import LookReason
from exp.offline_search.closed_loop import plugin


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


class Profile(V1):
    profile_version = 2

    def __init__(self, *args, design=None, blind_shadow=True, resample_p=1/16,
                 resample_draws=4, provenance=None, calibration_path="", **kwargs):
        super().__init__(*args, **kwargs)
        self.design_config = validate(design)
        self.nominal_p = self.p
        if self.design_config["episode_doses"] is not None and self.strata is not None:
            raise ValueError("episode dose mixture and absolute state strata are separate designs")
        if type(blind_shadow) is not bool or not 0 <= resample_p <= 1 or not 1 <= resample_draws <= 8:
            raise ValueError("invalid shadow sampling configuration")
        self.blind_shadow, self.resample_p, self.resample_draws = blind_shadow, resample_p, resample_draws
        self.provenance_config = provenance or {}
        self.calibration_path = calibration_path
        self.guard_calibration = None
        self.design = Design(self.design_config)
        self.commit_controls = 10

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        if not self.enabled:
            return
        awm = awm_of(self.base)
        deployed = lib if awm.cand_name == "current" else ctx.open_library(awm.cand_name)
        manifest_path = deployed.dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.catalog = dict(schema="r6p3.catalog.v2", model=ctx.model, cell=ctx.cell,
            base_spec=self.base_spec, base_kwargs=self.base_kwargs,
            base_fit=dict(path=self.base_fit, sha256=sha(self.base_fit)) if self.base_fit else None,
            guard_fit=dict(path=self.guard_fit, sha256=sha(self.guard_fit)) if self.guard_fit else None,
            interface=dict(action_valid_indices=list(range(7)), state_valid_indices=list(range(8)),
                H=int(awm.act.shape[1]), R=5, C=2, gripper_dim=6,
                closed_sign=1 if ctx.model == "pi05" else -1,
                action_units="model-normalized library units; RMS over valid coordinates",
                state_units="model-normalized; raw LIBERO eef xyz/axis-angle/fingers separately",
                camera_names=["observation/image", "observation/wrist_image"],
                preprocessing="examples.libero.main:flip both axes, resize_with_pad, uint8; model input transform",
                key_builder="deployed CP1 spatial pool16; raw float32 keys saved losslessly",
                derived_from="current LIBERO interface; not a universal robot declaration"),
            candidate_library=awm.cand_name, library_episode_ids=np.unique(awm.lib_ep).tolist(),
            fit_info=plugin._jsonable(getattr(self.base, "fit_info", {})),
            supplied=self.provenance_config,
            unavailable=dict(checkpoint_content_hash="supply provenance.checkpoint_sha256 from coordinator model manifest",
                             acquisition_cost="historical rejected/acquisition attempts not present in serving fit"))
        self.catalog["library_manifest"] = dict(path=str(manifest_path), sha256=sha(manifest_path), content=manifest)
        self.catalog["interface"].update(action_valid_indices=list(range(manifest["act_valid_dims"])),
            state_valid_indices=list(range(manifest["rs_valid_dims"])), H=manifest["H"], R=manifest["exec_steps"],
            C=len(manifest["img_source_keys"]) if manifest.get("img_source_keys") else len(list(deployed.dir.glob("key_v[0-9].npy"))),
            gripper_dim=manifest["gripper_dim"],
            derived_from="hash-addressed deployed library manifest, with explicit LIBERO wire adapter metadata")
        self.catalog["noise_floor"] = dict(status="not measured by prefit", reason="new same-state policy draws require model inference; deployment draws logged by resampling")
        # Lossless fitted arrays, including V7 isotonic calibration, are in the
        # hash-addressed source fit. Emit distance reference tables separately.
        self.catalog["loeo_distance_tables"] = {f"{t}:{r}": a.tolist() for (t, r), a in self.cdf.items()}
        guard = self.guard or self.base
        self.catalog["v7_calibration"] = plugin._jsonable(getattr(guard, "cal", {}))
        self.catalog["guard_fit_info"] = plugin._jsonable(getattr(guard, "fit_info", {}))
        if self.calibration_path:
            self.guard_calibration = json.loads(Path(self.calibration_path).read_text())
            expected = sha(self.guard_fit or self.base_fit)
            if self.guard_calibration["guard_fit_sha256"] != expected:
                raise ValueError("calibration artifact belongs to a different guard fit")
            self.catalog["episode_max_calibration"] = self.guard_calibration
            self.catalog["calibration_sha256"] = sha(self.calibration_path)
        payload = json.dumps(self.catalog, sort_keys=True, default=plugin._json_default).encode()
        self.catalog_sha256 = hashlib.sha256(payload).hexdigest()

    def reset(self, episode):
        super().reset(episode)
        self.design = Design(self.design_config)
        self.commit_controls = 10
        from .design import uniform
        doses = self.design_config["episode_doses"]
        self.p = self.nominal_p
        self.episode_assignment = dict(dose=self.p, probability=1., random=False)
        if doses is not None:
            u = uniform(self.seed, int(episode.task_id), int(episode.init), self.replicate, 0, "episode_dose")
            self.p = float(doses[min(int(u*len(doses)), len(doses)-1)])
            self.episode_assignment = dict(dose=self.p, probability=doses.count(self.p)/len(doses), uniform=u, random=True)

    def policy_tail_step(self, bq):
        if self.enabled and self.commit_controls == 5:
            return LookReason(8, "p3_randomized_policy_duration5")
        return super().policy_tail_step(bq)

    def blind_step(self, bq):
        if self.enabled and self.design_config["pre_guard"]:
            # A pre-guard experiment explicitly withholds B's forced CALL.
            # Commit that cache candidate for ten controls just like A;
            # otherwise B's no-progress look veto would shorten the treatment.
            return awm_of(self.base).blind_step(bq)
        return super().blind_step(bq)

    def query(self, q):
        if not self.enabled:
            return super().query(q)
        timings = {}
        original_base = self.base.query
        original_guard = self.guard.query if self.guard is not None else None
        def measured(key, fn):
            def invoke(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    timings[key] = (time.perf_counter()-start)*1000
            return invoke
        self.base.query = measured("base_query_ms", original_base)
        if self.guard is not None:
            self.guard.query = measured("shadow_guard_query_ms", original_guard)
        try:
            res = super().query(q)
        finally:
            self.base.query = original_base
            if self.guard is not None:
                self.guard.query = original_guard
        if self.enabled:
            self.record["timing"].update(timings)
            self.record["schema"] = "r6p3.anchor.v2"
            self.record["catalog_sha256"] = self.catalog_sha256
            self.record["episode_assignment"] = copy.deepcopy(self.episode_assignment)
            self.record["guards"]["base_confidence"] = float(res.confidence)
            guard = self.guard or self.base
            self.record["guards"]["controller_state_after_query"] = copy.deepcopy(getattr(guard, "_s", {}))
            self.record["guards"]["vision_progress"] = copy.deepcopy(getattr(guard, "_vision_progress", []))
            ex = self.record["guards"]["inputs_outputs"]
            ga = awm_of(guard)
            table = ga.tasks[int(q.task_id)]
            pos = np.searchsorted(table.rows, self.record["retrieval"]["rows"])
            modal = {}
            for camera, key in ((0, q.key_v0), (1, q.key_v1)):
                basis = getattr(ga, f"B{camera}T")
                projected = basis @ np.asarray(key, np.float32) - getattr(ga, f"muB{camera}")
                centered = projected - getattr(table, f"Vm{camera}")
                modal[f"vision_{camera}_pca64"] = projected
                modal[f"vision_{camera}_candidate_cosine"] = (getattr(table, f"V{camera}")[pos]
                    @ (centered/max(float(np.linalg.norm(centered)), 1e-12)))
            self.record["retrieval"]["modalities"] = modal
            self.record["guards"]["fired_tests"] = [i for i in range(6) if int(ex.get("os_flags", 0)) & (1 << i)]
            self.record["history"] = dict(hit=list(map(bool, q.hist_hit)),
                vision=list(map(bool, q.hist_has_vision)), previous_executed_chunk=None if q.prev_a_exec is None else np.array(q.prev_a_exec),
                decision_index=int(q.step), raw_state_finite=bool(np.isfinite(q.raw_state).all()))
            self.record["calibration"] = dict(distance_midrank=self.record["retrieval"]["d1_loeo_quantile"],
                family_level=None, dropped_tests=["coverage", "stuck", "lag", "overtime", "progress", "terminal"],
                reason="G episode-max simultaneous-test recipe is not the deployed guard; no calibrated family alpha claimed",
                reference="frozen-fit own-library row-LOEO distance plus source-fit V7 maps; see catalog")
            from .calibrate import statistics
            stats = statistics(guard, ex)
            self.record["calibration"]["statistics"] = stats
            if self.guard_calibration:
                tables = self.guard_calibration["per_task"].get(str(q.task_id), {})
                self.record["calibration"].update(dropped_tests=[k for k in stats if not tables.get(k)],
                    p_values={k: (1 + sum(x >= stats[k] for x in values)) / (1 + len(values))
                              for k, values in tables.items() if values},
                    p_value_scope="empirical episode-max reference, frozen-fit library policy histories; diagnostic only",
                    calibration_sha256=self.catalog["calibration_sha256"])
        return res


# V1's init wrapper resolves Engine at connection creation. Dispatch only the
# explicitly versioned class; a v1 fit loaded later in this process stays v1.
from . import hooks
from .v2_engine import Engine
if not getattr(hooks, "_v2_dispatch", False):
    _v1_engine = hooks.Engine
    def _dispatch(conn):
        cls = Engine if getattr(conn._osp_sessions[0].method, "profile_version", 1) == 2 else _v1_engine
        return cls(conn)
    hooks.Engine = _dispatch
    hooks._v2_dispatch = True
