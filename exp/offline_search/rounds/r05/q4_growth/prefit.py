"""The standard plugin prefit entry, suppressing even read-only git metadata."""
from exp.offline_search.closed_loop import plugin

if __name__ == '__main__':
    plugin._git_head = lambda: None  # The Q4 task explicitly prohibits git.
    raise SystemExit(plugin.prefit_main())
