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
    # No GPU reading on the server: the work-conserving pass gives it twice its desire, all on its own host.
    out = plan({'b': report('local', ('local', 'timan1'), 50)}, 1.)
    assert out['targets']['b'] == {'local': 100, 'timan1': 0}


def test_h100_fills_its_own_workers_first_like_the_4090():
    out = plan({'c': report('h100', ('h100', 'timan1'), 50)}, 1.)
    assert out['targets']['c'] == {'h100': 100, 'timan1': 0}


def saturation(server_cpu, gpu_util=60.):
    r = report('local', ('local', 'timan1'), 50)
    r['node_state']['local'].update(gpu_util=gpu_util, server_cpu=server_cpu)
    return sum(plan({'b': r}, 1.)['targets']['b'].values())


def test_cpu_bound_replicas_stop_the_work_conserving_doubling():
    assert saturation([104, 104, 103, 90]) == 50
    assert saturation([0, 102, 102, 102, 102, 102]) == 50
    assert saturation([40, 50, 60, 70]) == 100
    assert saturation([40, 50], gpu_util=95.) == 50
