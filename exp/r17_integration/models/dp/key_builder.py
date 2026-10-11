"""Native key builder reuses P1's stock collect/build/clear implementation."""
from exp.r17_integration.common.provider import NamedFieldsKeyBuilder


class DPKeyBuilder(NamedFieldsKeyBuilder):
    def __init__(self, adapter):
        super().__init__(adapter.metadata().feature_spec.fields)
