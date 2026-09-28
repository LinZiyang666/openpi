"""Opt-in LIBERO worker entry; coordinator only. Stock worker CLI is unchanged."""
import os


def main():
    from examples.libero import episode_runner, worker_entry
    from .snapshots import Client, run_episode
    directory = os.environ["P3_SNAPSHOT_DIR"]
    every = int(os.environ.get("P3_SNAPSHOT_EVERY", "1"))
    factory = episode_runner.default_client_factory
    original = episode_runner.LiberoEpisodeRunner

    class Runner(original):
        def __init__(self, *args, **kw):
            kw.update(client_factory=lambda endpoint: Client(factory(endpoint), directory, every),
                      run_episode_fn=run_episode)
            super().__init__(*args, **kw)

    episode_runner.LiberoEpisodeRunner = Runner
    worker_entry.main()


if __name__ == "__main__":
    main()
