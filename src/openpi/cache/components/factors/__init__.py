"""Verdict factor system: pluggable per-verdict descriptors.

Public surface is documented in `cache_system.md` §5.12. Concrete factor
classes live in submodules (runtime_continuity / source_window /
consensus); shared protocols, library statistics, and the registry live
in `base.py` and `registry.py`.

Importing `openpi.cache.components.factors.registry` triggers loading of
every concrete factor submodule so their `@register(...)` decorators run.
"""
