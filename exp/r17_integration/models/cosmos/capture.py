"""Minimal P6 teacher observer; emits no additional model products.

common.serving already detaches the actual FeatureSpec fields, state and
normalized chunk. LibraryDecisionSink binds the two received uint8 frames and
the execution/identity metadata. The post-MISS adapter hook therefore needs
no additional arrays, model calls, latent copies or random draws.
"""


class CosmosTeacherCaptureAdapter:
    def sample_fields(self, stage1, stage2, result):
        if result.snapshots:
            raise ValueError("Cosmos teacher observer does not support native snapshots")
        return {}
