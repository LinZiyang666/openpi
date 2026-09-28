"""V2 client entry; only the coordinator launches this module."""
import os
import sys


def main():
    from .client_compat import install_import_compat
    install_import_compat()
    from examples.libero import episode_runner, worker_entry
    from .telemetry import Client, run_episode
    factory = episode_runner.default_client_factory
    original = episode_runner.LiberoEpisodeRunner
    class Runner(original):
        def __init__(self, *args, **kw):
            kw.update(client_factory=lambda endpoint: Client(factory(endpoint), os.environ["P3_SNAPSHOT_DIR"],
                int(os.environ.get("P3_SNAPSHOT_EVERY", "1")), float(os.environ.get("P3_SNAPSHOT_P", "1"))),
                run_episode_fn=run_episode)
            super().__init__(*args, **kw)
    episode_runner.LiberoEpisodeRunner = Runner
    if "P3_ENV_SEED" in os.environ:
        sys.argv += ["--seed", str(int(os.environ["P3_ENV_SEED"]))]
    worker_entry.main()


if __name__ == "__main__":
    main()
