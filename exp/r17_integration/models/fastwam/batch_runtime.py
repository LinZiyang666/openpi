"""Default-off Fast-WAM LIBERO S07 runtime and composition ledger.

The immutable model manifest describes the default B1 policy. Batching is an
explicit execution study with its own identity; it does not assert byte parity
or policy-quality equivalence. No vendor/GPU imports at module import time.
"""
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
    numerics_policy: str = "native_bf16_batch_study_v1"

    def __post_init__(self):
        if type(self.max_batch) is not int or not 1 <= self.max_batch <= 32:
            raise ValueError("Fast-WAM max_batch must be an integer in 1..32")
        if isinstance(self.batch_wait_ms, bool) or not isinstance(self.batch_wait_ms, (int,float)) or not math.isfinite(self.batch_wait_ms) or not 0 <= self.batch_wait_ms <= 50:
            raise ValueError("Fast-WAM batch_wait_ms must be finite in 0..50")
        if self.numerics_policy != "native_bf16_batch_study_v1":
            raise ValueError("unknown batching numerics policy")

    @property
    def identity(self):
        return digest(asdict(self))

    def metadata(self):
        return {**asdict(self), "identity": self.identity, "stage_workers": [1,1,1],
                "default_off": True, "byte_identical_to_b1": False,
                "quality_equivalence": "unverified; matched batched control required",
                "hardware_qualification": "see BATCHING.md and measurement receipt"}


class BatchLedger:
    def __init__(self, path, config):
        self.path, self.config = Path(path), config
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock = threading.Lock()

    def write(self, row):
        line=json.dumps(row,sort_keys=True,allow_nan=False)+"\n"
        with self.lock, self.path.open("a") as handle:
            handle.write(line)

    def stage(self, stage, members):
        if any(member is None for member in members):
            raise ValueError('batched execution requires a semantic identity for every member')
        batch_id=uuid.uuid4().hex
        receipt=dict(schema="r17.fastwam.stage_batch.v1", batch_id=batch_id, stage=stage,
                     size=len(members), members=copy.deepcopy(members),
                     config_identity=self.config.identity, time=time.time())
        self.write(receipt)
        return [{**copy.deepcopy(receipt), "row": i} for i in range(len(members))]


class BatchRuntime:
    def __init__(self, adapter, config, ledger_path):
        from openpi.serving.batching_core import BatchingCore
        from .batcher import FastWAMStageBatcher
        if adapter.metadata().model_id != "fastwam_libero":
            raise ValueError("production batch study currently supports fastwam_libero only")
        for stage in (1,2,3):
            if os.environ.get(f"BATCHING_STAGE{stage}_WORKERS","1") != "1":
                raise ValueError("Fast-WAM batch study requires exactly one worker per stage")
        self.adapter, self.config = adapter, config
        self.previous_config,self.previous_ledger=adapter.batch_config,adapter.batch_ledger
        self.closed=False
        self.ledger=BatchLedger(ledger_path,config)
        # Production opt-in is distinct from the test-only vector bypass.
        adapter.batch_config=config
        adapter.batch_ledger=self.ledger
        self.core=BatchingCore(FastWAMStageBatcher(adapter),device=adapter.model.device,
                               max_batch_size=config.max_batch,max_wait_ms=config.batch_wait_ms)
        self.core.start()

    def close(self):
        if not self.closed:
            self.core.stop()
            self.adapter.batch_config,self.adapter.batch_ledger=self.previous_config,self.previous_ledger
            self.closed=True

    def wrap_connection_factory(self, base_factory):
        def factory(*args, **kwargs):
            connection=base_factory(*args, **kwargs)
            adapter=connection.adapter
            from .batcher import FastWAMBatchedAdapter
            if not isinstance(adapter,FastWAMBatchedAdapter) or adapter.coordinator is not self.core:
                raise ValueError("connection provider must be bound to this Fast-WAM batch runtime")
            original=connection.infer
            def infer(obs):
                adapter.last_batch_receipt={}
                result=original(obs)
                hit=result['__hit_meta__']
                batching=dict(config=self.config.metadata(), stages=copy.deepcopy(adapter.last_batch_receipt))
                hit['batching']=batching
                self.ledger.write(dict(schema="r17.fastwam.decision_batch.v1",time=time.time(),
                    connection_id=adapter.connection_id,decision_seq=hit['decision_seq'],source=hit['source'],
                    stage_calls=list(hit['stage_calls']),actual_nfe=hit['actual_nfe'],batching=batching))
                return result
            connection.infer=infer
            return connection
        return factory


def create_runtime(adapter, max_batch=1, batch_wait_ms=5, ledger_path=None):
    config=BatchConfig(max_batch,batch_wait_ms)
    if max_batch == 1:
        return None  # No queue, thread, file, metadata or model change by default.
    if ledger_path is None:
        raise ValueError("B>1 requires an explicit batch composition ledger path")
    return BatchRuntime(adapter,config,ledger_path)
