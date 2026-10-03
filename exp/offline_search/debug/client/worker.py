"""Python 3.8 stock worker with passive capture dependency injection.

Run only by the coordinator. P3 worker_v2 supplied this injection pattern.
The stock --seed (default 7) is authoritative and never arm-dependent.
"""
import os


def main():
    from .compat import install_import_compat
    install_import_compat()
    from examples.libero import episode_runner, worker_entry
    from .capture import Client, run_episode
    factory, original = episode_runner.default_client_factory, episode_runner.LiberoEpisodeRunner

    class Runner(original):
        def __init__(self, args, *pos, **kw):
            kw.update(client_factory=lambda endpoint: Client(
                factory(endpoint), os.environ["OSDEBUG_CLIENT_DIR"],
                campaign=os.environ.get("OSDEBUG_CAMPAIGN"), arm=os.environ.get("OSDEBUG_ARM"),
                env_seed=args.seed, oracle=os.environ.get("OSDEBUG_ORACLE", "0") == "1",
                oracle_radius=float(os.environ.get("OSDEBUG_ORACLE_RADIUS", ".10")),
                oracle_lift=float(os.environ.get("OSDEBUG_ORACLE_LIFT", ".03"))), run_episode_fn=run_episode)
            super().__init__(args, *pos, **kw)

    episode_runner.LiberoEpisodeRunner = Runner
    worker_entry.main()


if __name__ == "__main__":
    main()
