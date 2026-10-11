"""Run A's actual server, supplying its missing pure task table on H100.

The fallback is a read-only snapshot of exp/metaworld/tasks.py; no server logic
or evaluation convention is reimplemented here.
"""
import importlib.util
from pathlib import Path
import runpy
import sys
import types


def main():
    try:
        import exp.metaworld.tasks  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name not in ("exp.metaworld", "exp.metaworld.tasks"):
            raise
        module = types.ModuleType("exp.metaworld")
        module.__path__ = []
        sys.modules["exp.metaworld"] = module
        path = Path(__file__).parent / "shims/metaworld_tasks.py"
        spec = importlib.util.spec_from_file_location("exp.metaworld.tasks", path)
        tasks = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = tasks
        spec.loader.exec_module(tasks)
        module.tasks = tasks
    runpy.run_module("exp.mw_finetune.serve", run_name="__main__")


if __name__ == "__main__":
    main()
