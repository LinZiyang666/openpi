"""NVIDIA LIBERO N1.5 transforms adapted to one camera and four Sawyer channels."""
from .embodiment import register_embodiment

register_embodiment()

from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoColorJitter, VideoCrop, VideoResize, VideoToNumpy, VideoToTensor
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform

from .conventions import CHANNELS, HORIZON


class MetaWorldDataConfig(BaseDataConfig):
    video_keys = ["video.image"]
    state_keys = [f"state.{name}" for name in CHANNELS]
    action_keys = [f"action.{name}" for name in CHANNELS]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(HORIZON))

    def transform(self):
        return ComposedModalityTransform(transforms=[
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=.3, contrast=.4, saturation=.5, hue=.08),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={k: "min_max" for k in self.state_keys}),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={k: "min_max" for k in self.action_keys}),
            ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys,
                            action_concat_order=self.action_keys),
            GR00TTransform(state_horizon=1, action_horizon=HORIZON, max_state_dim=64, max_action_dim=32),
        ])
