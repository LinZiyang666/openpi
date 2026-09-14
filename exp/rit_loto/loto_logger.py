"""Server-side per-decision log for the closed-loop verification run.

Composition, not a copy: two transparent proxies sit between the production
``GrootCacheInterceptor`` and its runner / orchestrator. The runner proxy
remembers the last stage-1 output, the orchestrator proxy remembers the last
``check`` result and the chunk the interceptor *broadcast* -- which is the
chunk it executed, on every tier, so nothing is recomputed. After each
``get_action`` the logger cuts the stage-1 slices with the same function the
collector uses and writes one HDF5 step (vision_0/1, prompt_emb, robot_state,
``clean_action`` = executed chunk, no noise snapshots) plus one sidecar JSONL
row carrying the wire ``__hit_meta__`` and the decision identity. The
interceptor's return value is passed through untouched.

Every ``get_action`` clears both captures first and records only after the
interceptor returned; an exception leaves nothing behind for the next step to
reuse. Tensors are copied to CPU numpy before they are buffered so no GPU or
inference tensor outlives the call.

Environment: the GR00T island (torch, h5py; no jax).
"""

from __future__ import annotations

import json
import pathlib
import uuid
from typing import Any

import numpy as np
import torch

from openpi.cache.groot.interceptor import GrootCacheInterceptor
from openpi.cache.groot.key_builder import slice_groot_cp1_fields
from openpi.cache.types import VISION_0, VISION_1
from openpi.collect.data_collector import EpisodeDataCollector, InferenceEmbeddings

SIDECAR_NAME = "decisions.jsonl"
LIBERO_VISION_FIELDS = (VISION_0, VISION_1)


class _RecordingRunner:
    """Forwards everything to the staged runner; remembers the last ``run_stage1`` output."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_stage1: Any = None

    def reset(self) -> None:
        """Release the captured stage-1 output."""
        self.last_stage1 = None

    def run_stage1(self, *args: Any, **kwargs: Any) -> Any:
        """Capture the exact stage-1 result returned by the wrapped runner."""
        out = self._inner.run_stage1(*args, **kwargs)
        self.last_stage1 = out
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _RecordingOrchestrator:
    """Forwards everything to the orchestrator; remembers the last ``check`` and ``broadcast_action``."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_check: Any = None
        self.last_broadcast: Any = None

    def reset(self) -> None:
        """Release both captured dispatch values."""
        self.last_check = None
        self.last_broadcast = None

    def check(self, *args: Any, **kwargs: Any) -> Any:
        """Capture the production dispatch result without changing it."""
        out = self._inner.check(*args, **kwargs)
        self.last_check = out
        return out

    def broadcast_action(self, chunk: Any) -> Any:
        """Remember the executed chunk and forward its broadcast."""
        self.last_broadcast = chunk
        return self._inner.broadcast_action(chunk)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _to_np(chunk: Any) -> np.ndarray:
    t = torch.as_tensor(chunk).detach()
    if t.dim() == 3:
        if t.shape[0] != 1:
            raise ValueError(f"expected batch 1, got {tuple(t.shape)}")
        t = t[0]
    if t.dim() != 2:
        raise ValueError(f"expected [H, D] chunk, got {tuple(t.shape)}")
    out = t.to(device="cpu", dtype=torch.float32).contiguous()
    if out.is_inference():
        out = out.clone()
    return np.array(out.numpy(), dtype=np.float32, copy=True)


class GrootLotoLogger:
    """Cache-driven GR00T policy that also logs every decision to HDF5 + sidecar JSONL."""

    def __init__(
        self,
        policy: Any,
        runner: Any,
        *,
        orchestrator: Any,
        timer: Any,
        out_dir: str,
        experiment: str,
        run_tag: str,
        connection_id: str | None = None,
        vision_fields: tuple[str, ...] = LIBERO_VISION_FIELDS,
        identity_attrs: dict[str, Any] | None = None,
        h_exec: int = 5,
    ) -> None:
        if orchestrator is None:
            raise ValueError("GrootLotoLogger needs an orchestrator: it logs cache decisions")
        if not run_tag:
            raise ValueError("run_tag must be non-empty")
        self._runner = _RecordingRunner(runner)
        self._orch = _RecordingOrchestrator(orchestrator)
        self._interceptor = GrootCacheInterceptor(policy, self._runner, orchestrator=self._orch, timer=timer)
        self._connection_id = connection_id or uuid.uuid4().hex[:8]
        self._run_tag = str(run_tag)
        self._experiment = str(experiment)
        self._root = pathlib.Path(out_dir) / self._run_tag / f"conn_{self._connection_id}"
        self._root.mkdir(parents=True, exist_ok=True)
        self._collector = EpisodeDataCollector(str(self._root))
        self._sidecar = self._root / self._experiment / SIDECAR_NAME
        self._vision_fields = tuple(vision_fields)
        self._identity_attrs = dict(identity_attrs or {})
        self._h_exec = int(h_exec)
        self._state_index: torch.Tensor | None = None
        self._rows: list[dict] = []
        self._decision = 0
        self._task = ""
        self._task_id: int | None = None
        self._orig_init_state_idx: int | None = None
        self._episode_id = -1

    @property
    def connection_id(self) -> str:
        """The id stamped on every episode and directory of this connection."""
        return self._connection_id

    @property
    def root(self) -> pathlib.Path:
        """``<out_dir>/<run_tag>/conn_<id>``, where this connection's HDF5 and sidecar live."""
        return self._root

    # -- lifecycle -------------------------------------------------------

    def on_task_begin(self) -> None:
        """Forward the task boundary to the interceptor (and through it to the orchestrator)."""
        self._interceptor.on_task_begin()

    def on_task_end(self) -> None:
        """Forward the task end to the interceptor."""
        self._interceptor.on_task_end()

    def on_episode_start(
        self,
        experiment: str = "",
        task: str = "",
        episode_id: int = -1,
        episode_name: str = "",
        extra_metadata: dict | None = None,
    ) -> None:
        """Open an episode on the interceptor and the collector; stamp the frozen identity attrs."""
        md = dict(extra_metadata or {})
        self._task = str(task)
        self._episode_id = int(episode_id)
        self._task_id = int(md["task_id"]) if "task_id" in md else None
        self._orig_init_state_idx = int(md["orig_init_state_idx"]) if "orig_init_state_idx" in md else None
        self._rows = []
        self._decision = 0
        self._interceptor.on_episode_start(
            experiment=experiment, task=task, episode_id=episode_id, episode_name=episode_name,
            extra_metadata=extra_metadata,
        )
        self._collector.on_episode_start(
            experiment or self._experiment, task, episode_id, episode_name=episode_name, extra_metadata=extra_metadata,
        )
        schedule = self._runner.live_schedule()
        attrs = {
            "run_tag": self._run_tag,
            "connection_id": self._connection_id,
            "h_exec": self._h_exec,
            "denoise_schedule_id": schedule.schedule_id,
            "denoising_num_steps": int(schedule.num_steps),
            **self._identity_attrs,
        }
        for key, value in attrs.items():
            self._collector.set_episode_attr(key, value)

    def on_episode_end(self, success: bool) -> None:
        """Close the episode: interceptor hook, HDF5 flush, then the sidecar rows stamped with ``success``."""
        self._interceptor.on_episode_end(success)
        self._collector.on_episode_end(success=success)
        rows, self._rows = self._rows, []
        if rows:
            self._sidecar.parent.mkdir(parents=True, exist_ok=True)
            with self._sidecar.open("a", encoding="utf-8") as f:
                for row in rows:
                    row["episode_success"] = bool(success)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- inference -------------------------------------------------------

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Run the production interceptor, then log its stage-1 slices, hit meta and executed chunk.

        Captures are cleared before the call and again in ``finally``, so neither
        an exception nor the next step can see stale tensors; everything buffered
        is an owned CPU copy.
        """
        self._runner.reset()
        self._orch.reset()
        try:
            raw = self._interceptor.get_action(observations)
            stage1 = self._runner.last_stage1
            executed = self._orch.last_broadcast
            if stage1 is None:
                raise RuntimeError("interceptor returned without running stage 1; nothing to log")
            if executed is None:
                raise RuntimeError("interceptor returned without broadcasting the executed chunk; nothing to log")
            sliced = slice_groot_cp1_fields(
                stage1.input_embeds, stage1.image_token_mask, stage1.state, stage1.state_mask,
                enabled=None, expected_state_index=self._state_index, vision_fields=self._vision_fields,
            )
            if self._state_index is None:
                self._state_index = stage1.state_mask[0, -1].detach().clone()
            embs = InferenceEmbeddings(
                vision_embs=[np.array(sliced[name].detach().cpu().to(torch.float16).numpy(), copy=True)
                             for name in self._vision_fields],
                prompt_emb=np.array(sliced["prompt_emb"].detach().cpu().to(torch.float16).numpy(), copy=True),
                robot_state=np.array(sliced["robot_state"].detach().cpu().float().numpy(), dtype=np.float32, copy=True),
                noise_action_steps=[],
                clean_action=_to_np(executed),
                init_noise=None,
            )
        finally:
            self._runner.reset()
            self._orch.reset()
        meta = dict(raw.get("__hit_meta__") or {})
        self._collector.record_inference(embs)
        self._rows.append({
            "run_tag": self._run_tag,
            "connection_id": self._connection_id,
            "episode_id": self._episode_id,
            "decision_id": self._decision,
            "control_step_idx": self._decision * self._h_exec,
            "task": self._task,
            "task_id": self._task_id,
            "orig_init_state_idx": self._orig_init_state_idx,
            "s": meta.get("cp1_score"),
            "hit_type": meta.get("hit_type"),
            "start_t": meta.get("start_t"),
            "winner_id": meta.get("winner_id"),
            "searched": meta.get("searched"),
        })
        self._decision += 1
        return raw
