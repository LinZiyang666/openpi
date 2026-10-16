"""Coordinator-only R16 runner over the new filtered inventory/groups."""
from .vla import configure


def main():
    configure()
    from exp.offline_search.rounds.r16.k1ref.runner import main as run
    return run()


if __name__=='__main__':main()
