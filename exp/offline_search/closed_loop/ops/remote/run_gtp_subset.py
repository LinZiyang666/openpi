"""run_gtp restricted to some episode indices per task (smoke runs). OSCL_EPISODES="0" -> 10 episodes (ep_idx 0 of
every task); OSCL_TASKS="0,3" further restricts the tasks. The A-pool record / trials stay the full 50 so the pool
attestation is unchanged; only the planned TaskGraph shrinks. Pushed to the timan107 island (os_cl/)."""
import os
import sys

sys.path.insert(0, "/scratch/zixuans8/openpi_trace")

from exp.gate_threshold_pareto import run_gtp  # noqa: E402

KEEP_EP = {int(x) for x in os.environ["OSCL_EPISODES"].split(",") if x.strip()}
KEEP_T = {int(x) for x in os.environ.get("OSCL_TASKS", "").split(",") if x.strip()}
_orig = run_gtp.SweepStrategy._episodes


def _episodes(self, yaml_id, server):
    return [t for t in _orig(self, yaml_id, server)
            if t.episode_idx in KEEP_EP and (not KEEP_T or t.task_id in KEEP_T)]


run_gtp.SweepStrategy._episodes = _episodes

if __name__ == "__main__":
    run_gtp.main()
