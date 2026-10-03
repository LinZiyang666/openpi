"""T8 CLI: static episode HTML/PNG, captured images and physical timelines."""

from .cards import card, html_document
from .common import load_config, load_episodes, parser, report


def main(argv=None):
    args = parser("episode_card").parse_args(argv)
    episodes = load_episodes(args.run_root, args.arms, args.p3v2, args.limit)
    config = load_config(args, episodes)
    rows = [card(ep, args.out, config) for ep in episodes]
    report(
        args.out,
        "episode_card",
        {"episode_cards": rows},
        {"accepted_episodes": len(episodes)},
        episodes=episodes,
    )
    import html

    links = (
        "<ul>"
        + "".join(
            "<li><a href='"
            + html.escape(r["html"], quote=True)
            + "'>"
            + html.escape(str(r["arm"]) + ": " + str(r["episode_key"]))
            + "</a></li>"
            for r in rows
        )
        + "</ul>"
    )
    (args.out / "index.html").write_text(html_document("Episode cards", links))


if __name__ == "__main__":
    main()
