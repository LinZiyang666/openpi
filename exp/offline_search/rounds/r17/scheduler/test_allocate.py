from .allocate import plan

GIB = 2 ** 30


def state(node, cpu=40.):
    return dict(time=1., ram_available=200 * GIB, ram_floor=10 * GIB, vram_capacity=40000, cpu_capacity=cpu,
                workers_ram=0, worker_ram=GIB, worker_vram=150, worker_cpu=.25)


def report(server, nodes, desired, stable=0.):
    return dict(received=0., done=False, desired=desired, worker_type='metaworld', server_host=server,
                shared_stable_s=stable, active={}, node_state={n: state(n) for n in nodes},
                nodes={n: dict(max=500, **{'class': 'local' if n in ('local', 'h100') else 'shared'}) for n in nodes})


def test_4090_batch_takes_its_own_host_first_without_stability_wait():
    out = plan({'b': report('local', ('local', 'timan1'), 50)}, 1.)
    assert out['targets']['b'] == {'local': 50, 'timan1': 0}


def test_h100_local_workers_still_wait_for_a_stable_shared_pool():
    shared_first = plan({'c': report('h100', ('h100', 'timan1'), 50)}, 1.)
    assert shared_first['targets']['c'] == {'h100': 0, 'timan1': 50}
    stable = plan({'c': report('h100', ('h100', 'timan1'), 400, stable=700.)}, 1.)
    assert stable['targets']['c']['h100'] > 0
