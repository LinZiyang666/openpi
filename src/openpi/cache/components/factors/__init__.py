"""Verdict factor system: pluggable per-verdict descriptors.

Public surface is documented in `cache_system.md` §5.12. The 17 concrete
factors live in `online.py` (8 online: 4 desc × {action, state}),
`offline.py` (8 offline: 4 desc × {action, state}), and `topk.py`
(`topk_action_variance`). Shared protocols, dataclasses, and the
registry live in `base.py` and `registry.py`. The 4-layer architecture
also ships sibling `normalization/` (Layer 1), `calibrations/` (Layer 3),
and `composers/` (Layer 4) sub-packages.

Importing `openpi.cache.components.factors.registry` triggers loading of
every concrete factor submodule so their `@register(...)` decorators run.
"""
