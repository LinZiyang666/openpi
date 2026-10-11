"""Default-off DP-LIBERO runtime using stock S07 queues, private RNG and ledger."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from exp.r17_integration.common.contracts import digest


@dataclass(frozen=True)
class BatchConfig:
    max_batch: int = 1
    batch_wait_ms: float = 5
    numerics_policy: str = "dp_private_rng_vector_v1"

    def __post_init__(self):
        if type(self.max_batch) is not int or not 1 <= self.max_batch <= 32:
            raise ValueError("DP max_batch must be an integer in 1..32")
        if isinstance(self.batch_wait_ms, bool) or not isinstance(self.batch_wait_ms, (int, float)) or not math.isfinite(self.batch_wait_ms) or not 0 <= self.batch_wait_ms <= 50:
            raise ValueError("DP batch_wait_ms must be finite in 0..50")
        if self.numerics_policy != "dp_private_rng_vector_v1":
            raise ValueError("unknown DP batching numerics policy")

    @property
    def identity(self):
        return digest(asdict(self))

    def metadata(self):
        return {**asdict(self), "identity": self.identity, "stage_workers": [1, 1, 1],
                "default_off": True, "arm_steps": [100, 10, 5], "byte_identical_to_b1": False,
                "noise_policy": "per-request sequential B1 initial/posterior draws; independent of members/order",
                "language_policy": "frozen CPU B1 prompt embeddings memoized by exact text; 128-entry bound",
                "quality_equivalence": "requires matched-batch control SR; not established by throughput"}


class BatchLedger:
    def __init__(self, path, config):
        self.path, self.config = Path(path), config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, row):
        line = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        with self.lock, self.path.open("a") as stream:
            stream.write(line)

    def stage(self, stage, members):
        if any(member is None for member in members):
            raise ValueError("DP batch ledger requires a semantic identity for every member")
        receipt = dict(schema="r17.dp.stage_batch.v1", batch_id=uuid.uuid4().hex, stage=stage,
                       size=len(members), members=copy.deepcopy(members), config_identity=self.config.identity,
                       time=time.time())
        self.write(receipt)
        return [dict(copy.deepcopy(receipt), row=i) for i in range(len(members))]


class FrozenLanguageMemo:
    """Caches only the frozen, observation-independent B1 text function."""
    def __init__(self, language):
        from collections import OrderedDict
        self.language, self.values, self.lock = language, OrderedDict(), threading.RLock()

    def __call__(self, prompts):
        import torch
        rows = []
        with self.lock:
            for prompt in prompts:
                if prompt not in self.values:
                    self.values[prompt] = self.language([prompt]).detach().clone()
                    if len(self.values) > 128:
                        self.values.popitem(last=False)
                self.values.move_to_end(prompt)
                rows.append(self.values[prompt])
            return torch.cat(rows, 0).clone()


class BatchRuntime:
    def __init__(self, adapter, config, ledger_path):
        from openpi.serving.batching_core import BatchingCore
        if adapter.domain != "libero" or adapter.batch_diagnostic_label:
            raise ValueError("production DP batching requires the LIBERO adapter")
        if any(os.environ.get(f"BATCHING_STAGE{stage}_WORKERS", "1") != "1" for stage in (1, 2, 3)):
            raise ValueError("DP batching requires exactly one worker per stage; graph workspace is stage3-owned")
        self.adapter, self.config, self.closed = adapter, config, False
        self.previous_config, self.previous_ledger = adapter.batch_config, adapter.batch_ledger
        self.previous_language = adapter.language
        self.ledger = BatchLedger(ledger_path, config)
        adapter.batch_config, adapter.batch_ledger = config, self.ledger
        adapter.language = FrozenLanguageMemo(adapter.language)
        self.core = BatchingCore(adapter.make_stage_batcher(), device=adapter.device,
                                max_batch_size=config.max_batch, max_wait_ms=config.batch_wait_ms)
        try:
            self.core.start()
        except BaseException:
            adapter.batch_config, adapter.batch_ledger = self.previous_config, self.previous_ledger
            adapter.language = self.previous_language
            raise

    def close(self):
        if not self.closed:
            self.core.stop()
            self.adapter.batch_config, self.adapter.batch_ledger = self.previous_config, self.previous_ledger
            self.adapter.language = self.previous_language
            self.closed = True

    def wrap_connection_factory(self, base_factory):
        def factory(*args, **kwargs):
            connection = base_factory(*args, **kwargs)
            from .provider import CoordinatedAdapter
            adapter = connection.adapter
            if not isinstance(adapter, CoordinatedAdapter) or adapter.core is not self.core:
                raise ValueError("connection must be bound to this DP batch runtime")
            original = connection.infer
            def infer(obs):
                adapter.last_batch_receipt = {}
                result = original(obs)
                hit = result["__hit_meta__"]
                batching = dict(config=self.config.metadata(), stages=copy.deepcopy(adapter.last_batch_receipt))
                hit["batching"] = batching
                self.ledger.write(dict(schema="r17.dp.decision_batch.v1", time=time.time(),
                    connection_id=adapter.connection_id, decision_seq=hit["decision_seq"], source=hit["source"],
                    stage_calls=list(hit["stage_calls"]), actual_nfe=hit["actual_nfe"], batching=batching))
                return result
            connection.infer = infer
            return connection
        return factory


def create_runtime(adapter, max_batch=1, batch_wait_ms=5, ledger_path=None, *, force=False):
    config = BatchConfig(max_batch, batch_wait_ms)
    if max_batch == 1 and not force:
        return None  # Default path allocates no queue/thread/file and changes no model metadata.
    if ledger_path is None:
        raise ValueError("DP batched runtime requires a batch composition ledger")
    return BatchRuntime(adapter, config, ledger_path)
