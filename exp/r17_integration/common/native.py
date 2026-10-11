"""Append-only native contracts kept outside the sealed legacy modules."""
from dataclasses import dataclass
from typing import Any, Protocol
from openpi.cache.storage_types import CachePayload as LegacyCachePayload


@dataclass
class CachePayload(LegacyCachePayload):
    sampler_state: dict[str, Any] | None = None
    native_schedule_meta: dict[str, Any] | None = None


class NativeScheduleProvider(Protocol):
    def sampler_for_steps(self, num_steps: int) -> Any: ...
    def validate_snapshot(self, snapshot: Any, sampler: Any) -> None: ...


_providers = {}


def register_native_schedule_provider(name, provider):
    if name in _providers and _providers[name] is not provider:
        raise ValueError(f"native schedule provider already registered: {name}")
    _providers[name] = provider


def native_schedule_provider(name):
    try:
        return _providers[name]
    except KeyError:
        raise ValueError(f"unknown native schedule provider: {name}") from None
