"""Explicit client evidence shim over the unchanged P2 LIBERO runner."""
from contextlib import contextmanager

FIELDS = ('miss_mode','stall_status','stall_call','stall_extra_look','stall_fresh','stall_cooldown',
          'stall_anchor','executed_controls','corrector_enabled','corrector_applied',
          'corrector_delta_norm','corrector_computed','donor_rows','donor_weights','batching','resident_pid','resident_generation')


def extend_cost_row(base):
    def cost_row(result, identity, seq):
        row = base(result, identity, seq)
        if identity.get('cache_study') is True:
            hit = result.get('__hit_meta__', {})
            row.update({key:hit.get(key) for key in FIELDS})
            if 'knob_call' in hit:
                row['knob_call'] = hit['knob_call']
        return row
    return cost_row


@contextmanager
def evidence():
    from exp.r17_integration.benchmarks import execution
    original = execution.cost_row
    original_end = execution.end_episode
    execution.cost_row = extend_cost_row(original)
    def end_episode(client, kwargs, receipts):
        ack = original_end(client, kwargs, receipts)
        if receipts and receipts[-1].get('cache_study') is True:
            final = (ack or {}).get('execution')
            last = receipts[-1]
            if not isinstance(final, dict) or final.get('terminal') is not True or any(
                    final.get(k) != last.get(k) for k in ('decision_seq','executed_count','cursor_end')):
                raise ValueError('cache-study terminal execution ACK missing/mismatched')
            last.update(terminal=True, server_terminal_ack=final)
        return ack
    execution.end_episode = end_episode
    try:
        yield
    finally:
        execution.cost_row = original
        execution.end_episode = original_end


def main(argv=None):
    from exp.r17_integration.conductor.worker_entry import main as worker
    with evidence():
        worker(argv)


if __name__ == '__main__':
    main()
