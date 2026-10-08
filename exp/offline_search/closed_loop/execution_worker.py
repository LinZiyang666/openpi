"""Opt-in LIBERO worker with acknowledgments counted from completed env steps.

Shared LIBERO sources stay unchanged. Only R12 driver workers select this entry.
The proxy counts actual successful step returns after each inference, rather
than assuming a configured replan length was executed.
"""
from __future__ import annotations


def audited_run(run_episode, env, client, initial_state, task_description, args, max_steps, **kwargs):
    """Run the existing rollout with request-scoped, actual execution receipts."""
    if args.replan_steps != 5:
        raise ValueError('R12 audited worker requires replan_steps=5')
    state = dict(steps=0, decision=0, started=False, ok=True)

    class Environment:
        def __getattr__(self, name):
            return getattr(env, name)

        def step(self, action):
            try:
                result = env.step(action)
            except Exception:
                state['ok'] = False
                raise
            if state['started']:
                state['steps'] += 1
            return result

    class Client:
        def __getattr__(self, name):
            return getattr(client, name)

        def infer(self, observation):
            receipt = dict(executed_steps=state['steps'], exec_ok=state['ok'], decision_id=state['decision'])
            observation = dict(observation)
            extra = dict(observation.get('__extra__') or {})
            if any(k in extra for k in receipt):
                raise ValueError('R12 execution receipt already present; refusing conflicting writer')
            observation['__extra__'] = {**extra, **receipt}
            result = client.infer(observation)
            state.update(steps=0, decision=state['decision'] + 1, started=True, ok=True)
            return result

    return run_episode(Environment(), Client(), initial_state, task_description, args, max_steps, **kwargs)


def main():
    """Install the opt-in rollout wrapper before constructing the worker runner."""
    from examples.libero import main as rollout
    from examples.libero import worker_entry
    original = rollout._run_episode

    def run_episode(env, client, initial_state, task_description, args, max_steps, **kwargs):
        return audited_run(original, env, client, initial_state, task_description, args, max_steps, **kwargs)

    rollout._run_episode = run_episode
    worker_entry.main()


if __name__ == '__main__':
    main()
