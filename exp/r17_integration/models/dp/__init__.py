"""Opt-in DP-T integration. Importing this package does not load a model."""

INTERFACE_VERSION = "r17.contracts.v1"


def load_adapter(spec):
    from .loader import load_adapter as load
    return load(spec)


def register_provider(adapter, **kwargs):
    from .provider import register_provider as register
    return register(adapter, **kwargs)


def make_connection_factory(adapter, **kwargs):
    from .provider import make_connection_factory as make
    return make(adapter, **kwargs)


def make_stage_batcher(adapter):
    return adapter.make_stage_batcher()


def make_trace_adapter(adapter):
    return adapter.make_trace_adapter()
