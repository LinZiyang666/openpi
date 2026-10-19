"""MetaWorld MT50 task table, rollout constants and task-list export.

The table is RLinf's ``metaworld_config.json`` (prompt text verbatim, case kept;
``push-back-v3`` keeps the training-time label "Push the puck to a goal", not
the later LeRobot rewrite) with its four difficulty groups (easy 28 / medium 11
/ hard 6 / very hard 5). Row order is metaworld 3.0.0's ``MT50().train_classes``
order, and a task's position in it is its ``task_id``.

Episode identity is ``(task, idx, bench seed)``: ``idx`` indexes
``MT1(task, seed=BENCH_SEED).train_tasks`` (0..49) and is carried as the
conductor's ``orig_init_state_idx``, so every arm replays the same initial
state for the same identity.

Pure Python: importable by the driver (main venv) and the simulator venv alike.
Public interface: ``MT50_TASKS``, ``TASK_NAMES``, ``PROMPTS``, ``DIFFICULTY``,
the rollout constants, ``task_id_of`` and ``export_tasks``.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Rollout constants (RLinf standalone evaluation protocol)
# ------------------------------------------------------------------

BENCHMARK = "metaworld_mt50"
#: Seed of ``metaworld.MT1`` for every episode (a convention, equal to the LIBERO env seed).
BENCH_SEED = 7
#: Tasks sampled per environment by ``MT1`` (the valid ``idx`` range).
N_TRAIN_TASKS = 50
#: Zero-action steps after ``reset()`` before the first decision.
SETTLE_STEPS = 15
#: Policy steps per episode at most (settle steps excluded).
MAX_POLICY_STEPS = 160
#: Actions executed open-loop per inference call (= the checkpoint's action horizon).
REPLAN_STEPS = 5
CAMERA_NAME = "corner2"
CAMERA_ID = 2
CAMERA_POS = (0.75, 0.075, 0.7)
RENDER_SIZE = 480
#: Episodes per task of the formal run (idx 0..19).
EPISODES_PER_TASK = 20
DIFFICULTY_GROUPS = ("easy", "medium", "hard", "very_hard")

# ------------------------------------------------------------------
# Task table: (env name, prompt, difficulty) in MT50 train_classes order
# ------------------------------------------------------------------

MT50_TASKS: tuple[tuple[str, str, str], ...] = (
    ("assembly-v3", "Pick up a nut and place it onto a peg", "hard"),
    ("basketball-v3", "Dunk the basketball into the basket", "medium"),
    (
        "bin-picking-v3",
        "Grasp the puck from one bin and place it into another bin",
        "medium",
    ),
    ("box-close-v3", "Grasp the cover and close the box with it", "medium"),
    ("button-press-topdown-v3", "Press a button from the top", "easy"),
    (
        "button-press-topdown-wall-v3",
        "Bypass a wall and press a button from the top",
        "easy",
    ),
    ("button-press-v3", "Press a button", "easy"),
    ("button-press-wall-v3", "Bypass a wall and press a button", "easy"),
    ("coffee-button-v3", "Push a button on the coffee machine", "easy"),
    ("coffee-pull-v3", "Pull a mug from a coffee machine", "medium"),
    ("coffee-push-v3", "Push a mug under a coffee machine", "medium"),
    ("dial-turn-v3", "Rotate a dial 180 degrees", "easy"),
    ("disassemble-v3", "Pick a nut out of a peg", "very_hard"),
    ("door-close-v3", "Close a door with a revolving joint", "easy"),
    ("door-lock-v3", "Lock the door by rotating the lock clockwise", "easy"),
    ("door-open-v3", "Open a door with a revolving joint", "easy"),
    (
        "door-unlock-v3",
        "Unlock the door by rotating the lock counter-clockwise",
        "easy",
    ),
    ("hand-insert-v3", "Insert the gripper into a hole", "hard"),
    ("drawer-close-v3", "Push and close a drawer", "easy"),
    ("drawer-open-v3", "Open a drawer", "easy"),
    ("faucet-open-v3", "Rotate the faucet counter-clockwise", "easy"),
    ("faucet-close-v3", "Rotate the faucet clockwise", "easy"),
    ("hammer-v3", "Hammer a screw on the wall", "medium"),
    ("handle-press-side-v3", "Press a handle down sideways", "easy"),
    ("handle-press-v3", "Press a handle down", "easy"),
    ("handle-pull-side-v3", "Pull a handle up sideways", "easy"),
    ("handle-pull-v3", "Pull a handle up", "easy"),
    ("lever-pull-v3", "Pull a lever down 90 degrees", "easy"),
    (
        "pick-place-wall-v3",
        "Pick a puck, bypass a wall and place the puck",
        "very_hard",
    ),
    ("pick-out-of-hole-v3", "Pick up a puck from a hole", "hard"),
    ("pick-place-v3", "Pick and place a puck to a goal", "hard"),
    ("plate-slide-v3", "Slide a plate into a cabinet", "easy"),
    ("plate-slide-side-v3", "Slide a plate into a cabinet sideways", "easy"),
    ("plate-slide-back-v3", "Get a plate from the cabinet", "easy"),
    ("plate-slide-back-side-v3", "Get a plate from the cabinet sideways", "easy"),
    ("peg-insert-side-v3", "Insert a peg sideways", "medium"),
    ("peg-unplug-side-v3", "Unplug a peg sideways", "easy"),
    ("soccer-v3", "Kick a soccer into the goal", "medium"),
    ("stick-push-v3", "Grasp a stick and push a box using the stick", "very_hard"),
    ("stick-pull-v3", "Grasp a stick and pull a box with the stick", "very_hard"),
    ("push-v3", "Push the puck to a goal", "hard"),
    ("push-wall-v3", "Bypass a wall and push a puck to a goal", "medium"),
    ("push-back-v3", "Push the puck to a goal", "hard"),
    ("reach-v3", "Reach a goal position", "easy"),
    ("reach-wall-v3", "Bypass a wall and reach a goal", "easy"),
    ("shelf-place-v3", "Pick and place a puck onto a shelf", "very_hard"),
    ("sweep-into-v3", "Sweep a puck into a hole", "medium"),
    ("sweep-v3", "Sweep a puck off the table", "medium"),
    ("window-open-v3", "Push and open a window", "easy"),
    ("window-close-v3", "Push and close a window", "easy"),
)

TASK_NAMES: tuple[str, ...] = tuple(name for name, _, _ in MT50_TASKS)
PROMPTS: dict[str, str] = {name: prompt for name, prompt, _ in MT50_TASKS}
DIFFICULTY: dict[str, str] = {name: group for name, _, group in MT50_TASKS}


def task_id_of(name: str) -> int:
    """Canonical MT50 ``task_id`` of an environment name (``KeyError`` when unknown)."""
    try:
        return TASK_NAMES.index(name)
    except ValueError as exc:
        raise KeyError(f"unknown MetaWorld MT50 task {name!r}") from exc


def export_tasks(
    names: list[str] | None, episodes: int, init_offset: int = 0
) -> list[dict]:
    """Task list for ``exp.warm_reset.run prepare``: ``[{task_id, name, init_indices}]``.

    ``names`` selects environments (``None`` = all 50, in table order); ``init_indices``
    are the ``MT1.train_tasks`` indices ``init_offset .. init_offset + episodes - 1``,
    which must stay inside ``0 .. N_TRAIN_TASKS - 1``.
    """
    if episodes < 1 or init_offset < 0 or init_offset + episodes > N_TRAIN_TASKS:
        raise ValueError(f"init indices must lie in 0..{N_TRAIN_TASKS - 1}")
    selected = list(TASK_NAMES) if names is None else list(names)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("task names must be nonempty and unique")
    return [
        {
            "task_id": task_id_of(name),
            "name": name,
            "init_indices": list(range(init_offset, init_offset + episodes)),
        }
        for name in selected
    ]
