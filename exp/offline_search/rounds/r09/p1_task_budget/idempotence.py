"""Rebuild with frozen inputs and prove the owned scientific artifacts do not drift."""
import subprocess
import sys

from .common import HERE, RUN, sha


def main():
    files = [RUN / "arms.json", HERE / "task_tables.json", HERE / "source_lock.json",
             *sorted((RUN / "fits").glob("*.pkl"))]
    before = {str(p): sha(p) for p in files}
    result = subprocess.run([sys.executable, "-m", "exp.offline_search.rounds.r09.p1_task_budget.build"],
                            capture_output=True, text=True, check=True)
    print("\n".join(line for line in result.stdout.splitlines() if line.startswith(("P1_BUILD_OK", "TASKS", "FIT"))))
    assert before == {str(p): sha(p) for p in files}
    print("P1_IDEMPOTENCE_OK unchanged_arms=1 unchanged_tables=1 unchanged_source_lock=1 unchanged_wrapper_fits=6")


if __name__ == "__main__":
    main()
