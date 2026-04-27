"""Factor registry: name -> class, with explicit submodule load.

Concrete-factor submodules register themselves at import time via the
`@register(name)` decorator. Importing this module triggers loading of
every known submodule so the registry is populated as a side effect of
`from openpi.cache.components.factors import registry`.

Stub submodules MUST exist as importable modules at every batch (B0
ships them with the metadata layer + `extract` body raising
NotImplementedError; B1 / B2 fill in the algorithm bodies).
"""

from __future__ import annotations

from typing import Any

# Populated by submodules via @register decorator at import time.
_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator that records `name -> cls` in the global factor registry.

    Duplicate names raise ValueError immediately so config typos cannot
    silently shadow existing factor implementations.
    """

    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(
                f"Factor name {name!r} already registered to "
                f"{_REGISTRY[name].__qualname__}"
            )
        _REGISTRY[name] = cls
        return cls

    return deco


def get_class(name: str) -> type:
    """Return the registered class without instantiating.

    Used by config validation to introspect class-level capability flags
    (`requires_library_stats`, `requires_chain_walk`) and to call the
    `describe(params)` classmethod before any backend / library_stats
    is available.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown factor name {name!r}. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def build(name: str, **kwargs) -> Any:
    """Instantiate the registered class with `kwargs`."""
    return get_class(name)(**kwargs)


def known() -> set[str]:
    """Return the set of registered factor names (snapshot, not a view)."""
    return set(_REGISTRY)


# ------------------------------------------------------------------
# Force submodule imports so @register decorators run.
# Submodules ship in B0 as importable modules with full metadata; only
# their algorithm bodies (extract / compute_for_episode) raise
# NotImplementedError.
# ------------------------------------------------------------------

from openpi.cache.components.factors import (  # noqa: E402, F401
    consensus,
    runtime_continuity,
    source_window,
)
