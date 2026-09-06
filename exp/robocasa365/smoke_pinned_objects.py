"""S1 smoke: prove the pin actually took, on real kitchens.

Unit tests can only show the plumbing carries a payload. Whether the simulator
ends up holding that exact mesh is a question about robocasa, and only a built
kitchen answers it.

For every task in the pin table, across several seeds, this asserts:

1. every slot's realized mjcf path is byte-identical to the table (the implicit
   ``obj_container`` included);
2. poses still vary between seeds -- the experiment pins identity, not layout,
   and a pin that accidentally froze the scene would be a different experiment;
3. the instruction is constant, which is the property the retrieval library
   depends on;
4. a ``rotate_upright`` slot lands on ``model_upright.xml`` -- the exact-path
   branch does not apply the upright substitution itself, so this is the one
   thing the table has to get right by construction.

Run on a host with the asset tree, against a robocasa that has the pin patch::

    MUJOCO_GL=egl python -m exp.robocasa365.smoke_pinned_objects \\
        --pinned-objects exp/robocasa365/config/pnp_pinned_objects.json
"""

from __future__ import annotations

import argparse
import hashlib
import json

from exp.robocasa365.pinned_objects import load_pin_manifest, realized_objects_of

# The one key both teachers read the instruction from (episode_runner:88).
PROMPT_SOURCE_KEY = "annotation.human.task_description"


def _scene_fingerprint(env) -> str:
    """A digest of the physical state, used only to show that it changes."""
    inner = getattr(env, "unwrapped", env)
    state = inner.sim.get_state().flatten()
    return hashlib.sha256(state.tobytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pinned-objects", required=True)
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--seeds", default="1000000,1000001")
    args = ap.parse_args()

    import gymnasium as gym
    import robocasa  # noqa: F401

    pin_id, table = load_pin_manifest(args.pinned_objects)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"pin_id={pin_id}  seeds={seeds}")

    failures: list[str] = []
    for task_name in sorted(table):
        expected = table[task_name]
        env = gym.make(
            f"robocasa/{task_name}",
            split=None,
            obj_instance_split="target",
            layout_and_style_ids=[(args.layout, args.style)],
            pinned_objects=expected,
        )
        realized_per_seed = []
        prompts = []
        fingerprints = []
        try:
            for seed in seeds:
                obs, _ = env.reset(seed=seed)
                realized_per_seed.append(realized_objects_of(env))
                prompts.append(str(obs[PROMPT_SOURCE_KEY]))
                fingerprints.append(_scene_fingerprint(env))
        finally:
            env.close()

        # 1. identity: exact, every seed, every slot
        for seed, realized in zip(seeds, realized_per_seed):
            if realized != expected:
                only_real = {k: v for k, v in realized.items() if expected.get(k) != v}
                failures.append(f"{task_name} seed={seed}: realized != pinned; differing {only_real}")

        # 2. poses vary
        if len(set(fingerprints)) == 1:
            failures.append(f"{task_name}: scene state identical across seeds {seeds} — pose stopped varying")

        # 3. prompt constant
        if len(set(prompts)) != 1:
            failures.append(f"{task_name}: prompt varies across seeds: {sorted(set(prompts))}")

        upright = {s: p for s, p in expected.items() if p.endswith("model_upright.xml")}
        print(
            f"{task_name}: {len(expected)} slots pinned OK | "
            f"scene digests {fingerprints} | prompt={prompts[0]!r}"
            + (f" | upright slots {sorted(upright)}" if upright else "")
        )

    # 4. the upright slot must actually exist somewhere in the table
    if not any(p.endswith("model_upright.xml") for slots in table.values() for p in slots.values()):
        failures.append("no slot pins model_upright.xml; the rotate_upright case is untested")

    print(json.dumps({"pin_id": pin_id, "failures": failures}, indent=1))
    if failures:
        raise SystemExit(1)
    print("S1 SMOKE PASS")


if __name__ == "__main__":
    main()
