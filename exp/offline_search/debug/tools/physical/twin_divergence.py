"""E4 CLI: GR00T pairs must have a verified bit-identical physical prefix."""

from .paired_diverge import main as pair_main


def main(argv=None):
    pair_main(argv, twin=True)


if __name__ == "__main__":
    main()
