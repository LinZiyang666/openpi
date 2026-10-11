"""CPU native-kernel parity across episode-boundary process lifetimes."""
from dataclasses import replace
import json
import socket
import threading

import numpy as np
import pytest

from exp.r17_integration.benchmarks.libero_bridge import R17LiberoEpisodeRunner, decision_seed
from exp.r17_integration.ops.worker_lifetime import RECYCLE_EXIT, limited_loop
from exp.r17_integration.tests.benchmarks.test_libero import Env, TailClient, args, native_kernel, spec, task
from openpi.conductor import protocol
from openpi.conductor.worker import WorkerLoop


@pytest.mark.parametrize('seed,mode', [(7, 'paired_semantic'), (42, 'fixed'), (195, 'fixed')])
def test_recycling_preserves_original_inits_seed_requests_images_actions_and_attempts(tmp_path, seed, mode):
    sealed = replace(spec(), seed=seed, seed_mode=mode,
                     formal_indices=tuple(range(50)), library_indices=(), dev_indices=())
    tasks = [replace(task(sealed, global_id=tid, idx=idx), attempt=attempt)
             for tid, idx, attempt in [(3, 7, 1), (13, 49, 3), (23, 10, 2), (33, 0, 1)]]

    class Client(TailClient):
        def episode_start(self, **kwargs):
            self.cursor = self.seq = 0
            super().episode_start(**kwargs)

    class SeededEnv(Env):
        def __init__(self, environment_seed):
            super().__init__(done_at=6)
            self.environment_seed = environment_seed

    def run_fleet(lifetime):
        outputs, environments, clients, errors = [], [], [], []
        pending = list(tasks)
        while pending:
            client = Client(10)
            clients.append(client)
            def setup(dispatched):
                # Fresh native environments have this same explicit seed on
                # every setup; lifecycle count is never a seed or init index.
                environment = SeededEnv(sealed.seed)
                environments.append(environment)
                state = np.array([dispatched.task_id, dispatched.orig_init_state_idx], dtype=np.float64)
                return environment, state, sealed.tasks[dispatched.task_id][1], 220
            runner = R17LiberoEpisodeRunner(args(sealed), setup, spec=sealed,
                                           client_factory=lambda _: client, run_episode_fn=native_kernel())
            left, right = socket.socketpair()
            marker = tmp_path / f'recycle_{lifetime}_{len(clients)}.json'
            def driver():
                try:
                    for _ in range(min(lifetime, len(pending))):
                        kind, pull = protocol.recv_message(right)
                        assert kind == protocol.MSG_PULL and pull['worker_id'] == 'stable-worker'
                        assigned = pending.pop(0)
                        protocol.send_message(right, protocol.MSG_ASSIGN, {'task': protocol.task_to_wire(assigned)})
                        while True:
                            kind, payload = protocol.recv_message(right)
                            if kind == protocol.MSG_REPORT_RESULT:
                                assert payload.get('error') is None
                                outputs.append(payload)
                                break
                            assert kind == protocol.MSG_REPORT_PROGRESS
                    assert right.recv(1) == b''
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    right.close()
            thread = threading.Thread(target=driver, daemon=True)
            thread.start()
            loop = limited_loop(WorkerLoop, lifetime, marker)('stable-worker', tasks[0].server.key,
                                                              runner, connect=lambda: left)
            with pytest.raises(SystemExit) as stopped:
                loop.run_forever()
            assert stopped.value.code == RECYCLE_EXIT
            thread.join(timeout=5)
            assert not thread.is_alive() and not errors
            receipt = json.loads(marker.read_text())
            assert receipt['completed_episodes'] == lifetime
        return outputs, environments, [request for client in clients for request in client.requests]

    resident = run_fleet(4)
    recycled = run_fleet(1)
    assert [(r['task_uid'], r['attempt'], r['success'], r['n_steps']) for r in resident[0]] == [
        (r['task_uid'], r['attempt'], r['success'], r['n_steps']) for r in recycled[0]]
    for left, right, assigned in zip(resident[1], recycled[1], tasks, strict=True):
        assert left.closed and right.closed
        assert left.environment_seed == right.environment_seed == sealed.seed
        assert np.array_equal(left.initial, right.initial)
        assert left.initial.tolist() == [assigned.task_id, assigned.orig_init_state_idx]
        assert [a.tobytes() for a in left.actions] == [a.tobytes() for a in right.actions]
    assert len(resident[2]) == len(recycled[2])
    for left, right in zip(resident[2], recycled[2], strict=True):
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], np.ndarray):
                assert left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
                assert left[key].tobytes() == right[key].tobytes()
            else:
                assert left[key] == right[key]
        request = right['__request__']
        expected = sealed.seed if mode == 'fixed' else decision_seed(
            request['task_name'], request['original_init_idx'], request['decision_seq'], sealed.seed)
        assert request['explicit_seed'] == expected
