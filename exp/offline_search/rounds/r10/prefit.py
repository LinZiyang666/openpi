"""Local CPU-only standard plugin prefit; suppress repository revision inspection."""
from exp.offline_search.closed_loop import plugin

if __name__ == '__main__':
    plugin._git_head = lambda: None
    raise SystemExit(plugin.prefit_main())
