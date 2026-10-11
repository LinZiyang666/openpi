"""Explicit bounded startup qualification; reuses the one owned resident."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


def qualify(adapter, campaign, directory):
    if not campaign.get('smoke') or campaign['fastwam_batching']['batch_wait_ms'] != 5:
        raise ValueError('startup qualification supports explicit batched 5ms smoke only')
    import numpy as np
    from .batch_profile import BoardSampler, sweep
    path = Path(campaign['batch_qualification_observations'])
    with np.load(path, allow_pickle=False) as data:
        count = min(3, len(data['prompt']))
        observations = [{name: str(data[name][i]) if name == 'prompt' else data[name][i].copy()
                         for name in data.files} for i in range(count)]
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if '\n' in uuid:
        raise ValueError('startup qualification requires an unambiguous single GPU')
    old = (adapter.batch_config, adapter.batch_ledger, adapter.allow_unqualified_batch, adapter.phase_observer)
    receipt = Path(directory) / 'batch_qualification.json'
    try:
        with BoardSampler(uuid) as board:
            report = sweep(adapter, observations, deadline=time.monotonic()+120, board=board,
                           repeats=1, batch_sizes=(1,campaign['fastwam_batching']['max_batch']), receipt_path=receipt)
    finally:
        adapter.batch_config, adapter.batch_ledger, adapter.allow_unqualified_batch, adapter.phase_observer = old
    if report['status'] != 'passed' or any(row['status'] != 'passed' for row in report['concurrent']):
        raise ValueError('native B1/batch startup measurement did not pass')
    reference, batched = report['concurrent']
    if batched['numerics']['max_abs'] > .03125:
        raise ValueError('batch normalized action deviation exceeds two bf16 increments at magnitude2')
    import hashlib
    summary = dict(status='PASS', receipt=str(receipt), receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        recorded_observations=str(path), observations_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        max_batch=batched['max_batch'], batch_wait_ms=batched['batch_wait_ms'],
        decisions_per_second=batched['decisions_per_second'],
        b1_decisions_per_second=reference['decisions_per_second'],
        throughput_multiplier=batched['decisions_per_second']/reference['decisions_per_second'],
        numerics=batched['numerics'], mean_actual_batch_size=batched['mean_actual_batch_size'],
        quality_equivalence='four-episode smoke is not success-rate equivalence evidence',
        board_power_scope='whole GPU including preexisting Cosmos; not Fast-WAM-only power')
    (Path(directory)/'BATCH_QUALIFICATION.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
