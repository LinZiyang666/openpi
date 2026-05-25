"""Interface-side C2 auto-guard for VectorStoreBackend.

Verifies that ``VectorStoreBackend.__init_subclass__`` wraps every subclass
mutation method with the runtime write-frozen guard, so any pluggable backend
gets C2 (``BackendFrozenError`` post-``freeze()``) without writing a single
``_check_frozen`` call. These tests deliberately define backends that contain
NO frozen-check boilerplate — the guard must come entirely from the ABC.
"""

from __future__ import annotations

import pytest

from openpi.cache.backend_base import BackendFrozenError, VectorStoreBackend


class _BareBackend(VectorStoreBackend):
    """Minimal backend with zero ``_check_frozen`` calls anywhere.

    Implements only the abstract surface plus ``insert`` / ``delete``. It does
    NOT override ``batch_insert`` (so it inherits the base default) and defines
    no ``load_artifact`` / ``upsert``.
    """

    def __init__(self) -> None:
        self.inserted: list = []
        self.deleted: list = []

    @property
    def vector_dims(self) -> dict[str, int]:
        return {"v": 4}

    def supported_filters(self) -> frozenset[str]:
        return frozenset()

    def insert(self, entry) -> None:
        self.inserted.append(entry)

    def search(self, spec):
        return []

    def fetch_payload(self, id):
        raise KeyError(id)

    def delete(self, ids) -> None:
        self.deleted.extend(ids)

    def count(self) -> int:
        return len(self.inserted)


# ------------------------------------------------------------------
# T1 — bare subclass auto-guarded (core acceptance)
# ------------------------------------------------------------------


def test_bare_subclass_auto_guarded_without_any_check_frozen():
    b = _BareBackend()
    # Pre-freeze: mutations succeed.
    b.insert("e1")
    b.delete(["e1"])
    assert b.is_frozen is False

    b.freeze()
    assert b.is_frozen is True

    # Post-freeze: insert / delete raise — enforcement came purely from the ABC.
    with pytest.raises(BackendFrozenError):
        b.insert("e2")
    with pytest.raises(BackendFrozenError):
        b.delete(["e1"])


# ------------------------------------------------------------------
# T2 — inherited base batch_insert fail-fasts at the entry
# ------------------------------------------------------------------


def test_inherited_batch_insert_guarded_at_entry():
    b = _BareBackend()
    b.freeze()
    # The base default batch_insert self-guards at its head, so it raises even
    # for an empty list (i.e. before any per-entry insert loop).
    with pytest.raises(BackendFrozenError):
        b.batch_insert([])


# ------------------------------------------------------------------
# T3 — wrapper preserves introspection metadata
# ------------------------------------------------------------------


def test_guard_preserves_introspection():
    assert _BareBackend.insert.__name__ == "insert"
    # functools.wraps links the guard wrapper back to the original method.
    assert hasattr(_BareBackend.insert, "__wrapped__")


# ------------------------------------------------------------------
# T4 — inheritance is guarded exactly once (idempotent)
# ------------------------------------------------------------------


def test_inheritance_guarded_once():
    class _Child(_BareBackend):
        pass

    # _Child does not redefine insert → it inherits the parent's already-wrapped
    # method and is NOT re-wrapped (no entry in its own __dict__).
    assert "insert" not in _Child.__dict__

    c = _Child()
    c.freeze()
    with pytest.raises(BackendFrozenError):
        c.insert("x")

    class _ChildOverride(_BareBackend):
        def insert(self, entry) -> None:
            super().insert(entry)

    co = _ChildOverride()
    co.insert("ok")  # pre-freeze: child guard + parent guard both no-op
    assert co.inserted == ["ok"]
    co.freeze()
    with pytest.raises(BackendFrozenError):
        co.insert("x")


# ------------------------------------------------------------------
# T5 — native batch_insert override is auto-guarded (mirrors Qdrant)
# ------------------------------------------------------------------


def test_native_batch_insert_override_guarded():
    class _NativeBatch(_BareBackend):
        def batch_insert(self, entries):
            self.inserted.extend(entries)
            return None

    b = _NativeBatch()
    b.batch_insert(["a", "b"])  # pre-freeze OK
    assert b.inserted == ["a", "b"]
    b.freeze()
    with pytest.raises(BackendFrozenError):
        b.batch_insert(["c"])


# ------------------------------------------------------------------
# T6 — a subclass can extend _MUTATION_METHODS to guard extra entries
# ------------------------------------------------------------------


def test_subclass_can_extend_mutation_methods():
    class _ExtraMut(_BareBackend):
        _MUTATION_METHODS = (*_BareBackend._MUTATION_METHODS, "wipe")

        def wipe(self) -> None:
            self.inserted.clear()

    b = _ExtraMut()
    b.inserted.append("x")
    b.wipe()  # pre-freeze OK
    assert b.inserted == []
    b.freeze()
    with pytest.raises(BackendFrozenError):
        b.wipe()


# ------------------------------------------------------------------
# T7 — a @functools.wraps(Parent.insert) override stays guarded (G2 R1)
# ------------------------------------------------------------------


def test_wraps_parent_override_is_still_guarded():
    """A subclass override decorated with ``functools.wraps(Parent.insert)``
    must still be guarded. ``functools.wraps`` copies the parent wrapper's
    ``__dict__``, so ``__init_subclass__`` must NOT skip wrapping based on a
    copied marker — it must always wrap subclass-defined mutation methods.
    """
    import functools

    class _WrapsOverride(_BareBackend):
        @functools.wraps(_BareBackend.insert)
        def insert(self, entry) -> None:
            self.inserted.append(("wrapped", entry))

    b = _WrapsOverride()
    b.insert("ok")  # pre-freeze OK
    assert b.inserted == [("wrapped", "ok")]
    b.freeze()
    with pytest.raises(BackendFrozenError):
        b.insert("x")
