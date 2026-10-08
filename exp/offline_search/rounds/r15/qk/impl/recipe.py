"""Existing R11 recipes in an explicitly student-feature coordinate system."""
import numpy as np
from exp.offline_search.rounds.r11.knob.recipe import R11Knob


class FeatureGeometry:
    """Internal compatibility adapter. Its keys are student features, never tokens."""
    def __init__(self, look):
        self.look = look
    def __getattr__(self, name):
        return getattr(self.look, name)
    key_v0 = property(lambda s: s.look.light_features[0])
    key_v1 = property(lambda s: s.look.light_features[1])
    hist_key_v0 = property(lambda s: s.look.hist_light_features[:, 0])
    hist_key_v1 = property(lambda s: s.look.hist_light_features[:, 1])
    has_vision = property(lambda s: s.look.has_look)
    hist_has_vision = property(lambda s: s.look.hist_has_look)


class LightKeyKnob(R11Knob):
    family = 'r15_light_symmetric'
    def __init__(self, library='bpool_cs', episode_subset=None, method='off', target=None, encoder_spec=None):
        super().__init__(library, episode_subset, method, target)
        self.light_key_spec = encoder_spec
    def fit(self, lib, ctx):
        raise ValueError('light serving requires a complete prebuilt library-only artifact')
    def query(self, q):
        return super().query(FeatureGeometry(q))
