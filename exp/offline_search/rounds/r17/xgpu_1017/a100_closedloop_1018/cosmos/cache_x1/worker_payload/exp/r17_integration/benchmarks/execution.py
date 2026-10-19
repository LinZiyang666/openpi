"""Client-only execution/timing receipts shared by the owned benchmark adapters."""
from __future__ import annotations


def instrument_model_lock(adapter):
    """Opt-in bridge for a connection-local adapter with a shared model_lock.

    Retain the underlying lock and its exclusion semantics; measure acquisition
    wait in a local view. P1 can consume the standard drain hook. No model method,
    resident lock object, native tensor, RNG, or global provider is modified.
    """
    if callable(getattr(adapter, 'drain_queue_wait_ms', None)):
        return adapter
    import time
    native_lock = adapter.model_lock
    class TimedLock:
        wait_ms = 0.0
        def acquire(self, *args, **kwargs):
            begin = time.perf_counter()
            acquired = native_lock.acquire(*args, **kwargs)
            self.wait_ms += (time.perf_counter() - begin) * 1000.
            return acquired
        def release(self):
            return native_lock.release()
        def __enter__(self):
            self.acquire()
            return self
        def __exit__(self, *args):
            self.release()
        def __getattr__(self, name):
            return getattr(native_lock, name)
        def drain(self):
            wait, self.wait_ms = self.wait_ms, 0.0
            return wait
    measured = TimedLock()
    adapter.model_lock = measured
    adapter.drain_queue_wait_ms = measured.drain
    return adapter


def ack_envelope(receipt):
    return {k: receipt[k] for k in ('decision_seq', 'executed_count', 'terminal')}


def end_episode(client, kwargs, receipts):
    """Use the R17 client control; never omit a final decision acknowledgment."""
    if not receipts:
        return client.episode_end(**kwargs)
    ack = ack_envelope(receipts[-1])
    ack['terminal'] = True
    # Original client controls accept arbitrary kwargs via _control; preserve
    # the original socket/transport without modifying frozen client files.
    if hasattr(client, '_send_ctrl'):
        return client._send_ctrl({'__ctrl__': 'episode_end', '__success__': kwargs.get('success', False),
                                  '__execution__': ack}, expected_ack='episode_end')
    return client.episode_end(**kwargs, execution=ack)


def cost_row(result, identity, seq):
    hit = result.get('__hit_meta__', {})
    timing = result.get('server_timing', {})
    row = {**identity, 'decision_seq': seq,
           **{k: hit.get(k) for k in ('source', 'cursor', 'stage_calls', 'actual_nfe', 'sampler_identity',
                                      'commit_horizon', 'cost_ms', 'stage_timing_ms', 'stage_ms', 'capture_enabled', 'pricing_eligible',
                                      'queue_wait_ms', 'service_ms', 'stage_queue_wait_ms', 'stage_service_ms',
                                      'gpu_ms', 'gpu_service_ms', 'queue_wait_status', 'pricing_time_field', 'service_timing_scope', 'concurrency',
                                      'ir_calibration', 'allow_lossy_sampler', 'sampler_study_label', 'sampler_quality_status',
                                      'capture_overhead_ms', 'capture_sink_ms', 'timing_kind', 'bundle_identity', 'recipe_identity')},
           'server_timing': dict(timing), 'timing_scope': 'r17.policy'}
    row['timing_scope'] = hit.get('timing_scope', 'r17.policy')
    if row['cost_ms'] is None:
        row['cost_ms'] = timing.get('infer_ms')
        row['timing_scope'] = 'server_timing.infer_ms' if row['cost_ms'] is not None else 'unavailable'
    if row['service_ms'] is None or row['queue_wait_ms'] is None:
        # Preserve latency diagnostics without manufacturing model service time.
        row['pricing_eligible'] = False
    if row['queue_wait_status'] is None:
        row['queue_wait_status'] = 'measured' if row['queue_wait_ms'] is not None else 'unsupported'
    return row
