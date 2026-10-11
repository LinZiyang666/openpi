"""Opt-in Fast-WAM hooks. Importing this package does not register a provider."""

from .loader import LoadSpec, load_adapter
from .provider import make_connection_factory, register_provider

__all__ = ["LoadSpec", "load_adapter", "make_connection_factory", "register_provider"]
