"""CPU checks for read-only accounting and worker generation fences."""
import pytest

from exp.r17_integration.diagnostics.libero_memory import compare, kib_fields, residual_bytes, snapshot


def test_residual_does_not_double_count_reclaimable_shmem_or_vmalloc():
    fields = kib_fields('MemTotal: 1000 kB\nMemFree: 100 kB\nAnonPages: 200 kB\n'
                        'Cached: 50 kB\nBuffers: 10 kB\nSlab: 40 kB\n'
                        'Shmem: 20 kB\nKReclaimable: 30 kB\nSReclaimable: 30 kB\n'
                        'VmallocUsed: 9000 kB\n')
    assert residual_bytes(fields) == 600 * 1024
    assert residual_bytes(dict(fields, KReclaimable=45 * 1024)) == 585 * 1024


def test_exact_campaign_and_module_match_no_import_or_process_mutation(tmp_path):
    (tmp_path / 'meminfo').write_text('MemTotal: 1000 kB\nMemFree: 100 kB\nAnonPages: 200 kB\n')
    for pid, module, worker in [(10, 'exp.r17_integration.ops.worker_lifetime', 'owned_node_0'),
                                (11, 'exp.r17_integration.conductor.worker_entry', 'owned_other_node_0'),
                                (12, 'exp.r17_integration.conductor.worker_entry', 'foreign_node_0'),
                                (13, 'other.entry', 'owned_node_1')]:
        proc = tmp_path / str(pid)
        proc.mkdir()
        (proc / 'cmdline').write_bytes(('python\0-m\0' + module + '\0--worker-id\0' + worker + '\0').encode())
        (proc / 'stat').write_text(str(pid) + ' (python (worker)) S ' + ' '.join(['0'] * 18 + ['12345']))
        (proc / 'status').write_text('VmRSS: 256 kB\nVmLck: 0 kB\nVmPin: 0 kB\n')
        (proc / 'smaps_rollup').write_text('Pss: 240 kB\nLocked: 0 kB\n')
        (proc / 'maps').write_text('000-100 rw-p /dev/nvidia0\n')
        (proc / 'fd').mkdir()
        (proc / 'fd/0').symlink_to('/dev/nvidia0')
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    data = snapshot('owned', proc_root=tmp_path)
    assert [w['pid'] for w in data['workers']] == [10]
    assert data['workers'][0]['start_ticks'] == '12345'
    assert data['workers'][0]['smaps_bytes'] == dict(Pss=240 * 1024, Locked=0)
    assert data['workers'][0]['nvidia_fds'] == data['workers'][0]['nvidia_maps'] == 1
    assert not data['errors']
    assert before == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_growth_is_per_dispatch_without_dividing_by_worker_count_twice():
    before = dict(campaign='owned', hostname='node', time=1., residual_bytes=100,
                  meminfo_bytes=dict(AnonPages=200), workers=[dict(pid=1, start_ticks='2'), dict(pid=2, start_ticks='3')])
    after = dict(before, time=11., residual_bytes=300)
    delta = compare(before, after, dispatched_episodes=4)
    assert delta['estimated_driver_bytes_per_worker_episode'] == 50
    assert delta['anon_delta_bytes'] == 0
    with pytest.raises(ValueError, match='generations'):
        compare(before, dict(after, workers=[dict(pid=3, start_ticks='4')]), dispatched_episodes=4)
    with pytest.raises(ValueError, match='positive'):
        compare(before, after, dispatched_episodes=0)
