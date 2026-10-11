"""Baseline synchronous R17 driver with the cache study's bundle/group planner."""
import json
import os
from pathlib import Path
import time

from exp.r17_integration.qualification.driver import ProbeRecordingDriver


class SynchronousEvidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.failure = None
        self.rows = {}

    def submit(self, records):
        try:
            for name, rows in records.items():
                with (self.directory/name).open('a') as f:
                    for row in rows:
                        f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                        f.flush()
                        os.fsync(f.fileno())
                        self.rows[name] = self.rows.get(name, 0)+1
        except Exception as exc:
            self.failure = repr(exc)
            raise

    def close(self):
        return dict(schema='r17.synchronous_evidence.v1', status='PASS' if self.failure is None else 'FAIL',
                    mode='legacy', per_row_fsync=True, rows=dict(self.rows), error=self.failure)


class LegacyCacheDriver(ProbeRecordingDriver):
    """Same scheduler/fencing/per-row fsync as formal_driver's K5 baseline."""
    def __init__(self, strategy, *, evidence_directory, live_fleets=False, **kwargs):
        self.evidence = SynchronousEvidence(evidence_directory)
        def steps(stage, rows):
            self.evidence.submit({'per_step.jsonl': rows})
        super().__init__(strategy, per_step_writer=steps, **kwargs)

    def handle_pull(self, key):
        kind, payload = super().handle_pull(key)
        if isinstance(payload, dict) and isinstance(payload.get('task'), dict):
            task = payload['task']
            self.evidence.submit({'worker_dispatch.jsonl': [dict(
                worker_id=getattr(self._shutdown_local, 'worker_id', None), task_uid=task['task_uid'],
                attempt=task.get('attempt', 1), time=time.time())]})
        return kind, payload

    def handle_result(self, result):
        with self._scheduler._lock:
            uid = result.get('task_uid')
            entry = self._scheduler._ep_index.get(uid)
            accepted = entry is not None and uid in self._scheduler._rt[entry[0]].dispatched and (
                result.get('attempt') == self._scheduler._dispatch_gen.get(uid))
            super().handle_result(result)
            self.evidence.submit({'attempts.jsonl': [dict(
                {k:v for k,v in result.items() if k != 'per_step_rows'}, accepted=accepted,
                worker_id=getattr(self._shutdown_local, 'worker_id', None), time=time.time())]})

    def close(self):
        try:
            return self.evidence.close()
        finally:
            super().close()
