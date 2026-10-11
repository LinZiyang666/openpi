"""Opt-in GR00T N1.5 MetaWorld integration; no vendor or GPU imports."""

def register_provider(provider=None, **kwargs):
    from .provider import register_provider as register
    return register(provider, **kwargs)


def load_adapter(spec, **kwargs):
    from .loader import load_adapter as load
    return load(spec, **kwargs)


def make_connection_factory(*args, **kwargs):
    from .serve import make_connection_factory as make
    return make(*args, **kwargs)


def make_stage_batcher(adapter):
    return adapter.make_stage_batcher()


def make_trace_adapter(*args, **kwargs):
    from .trace import make_trace_adapter as make
    return make(*args, **kwargs)
