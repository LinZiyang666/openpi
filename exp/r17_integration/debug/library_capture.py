"""Opt-in teacher allowlist, over the existing DecisionSink/BlockWriter path."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading

import numpy as np

from .capture_bridge import CaptureBridge, DecisionSink
from .validate import EPISODE_KEYS, digest, model_binding

IMAGE_KEYS = ('observation/image', 'observation/wrist_image')


def camera_keys(cameras):
    return IMAGE_KEYS[:len(cameras)]


def campaign_capture_factory(campaign, directory):
    """Same local/H100 teacher path for ordinary and batched residents."""
    options = dict(capture_images=campaign.get('capture_images', True))
    if campaign.get('server_node') == 'h100':
        from exp.r17_integration.ops.library_staging import admit, ShardBudget
        from .library_shards import ShardedLibraryCaptureFactory
        admit(campaign)
        return ShardedLibraryCaptureFactory(directory, ShardBudget(campaign), **options)
    return LibraryCaptureFactory(directory, **options)


RECORD_FIELDS = set(EPISODE_KEYS) | {
    'decision_seq', 'episode_id', 'library_episode_id', 'task_id', 'suite_task_id',
    'init_pool', 'init_state_sha256', 'source', 'cursor', 'executed_count', 'execution_status',
    'stage_calls', 'actual_nfe', 'sampler_identity', 'split', 'seed', 'task_table_identity',
    'model_identity', 'norm_identity', 'layout_identity', 'image_sources', 'trial',
}


class LibraryCaptureBridge(CaptureBridge):
    def capture(self, record, arrays, *, rawkeys=None, **kwargs):
        if not self.enabled:
            return None
        # Filtering precedes numeric copying/serialization. Poisoned unwanted
        # products can never enter NPZ, H5, queues, or a post-collection cleanup.
        selected = {name:arrays[name] for name in ('normalized_actions', 'state')}
        if self.capture_images:
            for i in range(len(self.binding['model']['feature_spec']['real_cameras'])):
                selected['image_'+str(i)] = arrays['native%2Fimage_'+str(i)]
        fields = self.binding['model']['feature_spec']['fields']
        keys = {name:rawkeys[name] for name in fields}
        row = {key:value for key,value in record.items() if key in RECORD_FIELDS}
        if row.get('split') != 'library' or (row.get('init_pool') != 'B' and row.get('seed') != 8):
            raise ValueError('teacher capture requires library provenance')
        row['episode_id'] = row['library_episode_id']
        return super().capture(row, selected, rawkeys=keys, prompt='')


class LibraryDecisionSink(DecisionSink):
    def __init__(self, bridge, episode_metadata, *, image_keys=IMAGE_KEYS, capture_images=None):
        super().__init__(bridge, episode_metadata)
        self.image_keys = image_keys
        self.capture_images = getattr(bridge, 'capture_images', True) if capture_images is None else capture_images
        self._images = None

    def bind_observation(self, observation):
        """Retain the exact uint8 wire bytes passed to this connection's model.

        No resizing/camera synthesis. Model-specific internal encoder transforms
        remain model-owned; image_sources explicitly identifies this cut point.
        """
        if self.bridge is not None and self.bridge.enabled:
            from .attempts import decision_intent
            metadata = self.episode_metadata(None) if callable(self.episode_metadata) else self.episode_metadata
            decision_intent(self.bridge.directory, metadata)
        images = []
        for key in self.image_keys if self.capture_images else ():
            value = np.asarray(observation[key])
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError('teacher images must be real uint8 HWC RGB wire images')
            images.append(value.copy(order='C'))
        self._images = tuple(images)

    def __call__(self, capture):
        if not self.bridge.enabled:
            return
        if self._images is None:
            raise ValueError('teacher capture has no bound model-input images')
        sampler = self.bridge.binding['sampler']
        if capture.source != 'policy' or capture.actual_nfe != sampler['num_steps']:
            raise ValueError('teacher must be cache-off full native sampler at every decision')
        fields = self.bridge.binding['model']['feature_spec']['fields']
        minimal = replace(capture, fields={**{name:capture.fields[name] for name in fields},
                                         'robot_state':capture.fields['robot_state']},
            native_snapshots=(), diagnostics={}, sampler_receipt={'identity':digest(sampler)},
            model_capture={'image_'+str(i):value for i,value in enumerate(self._images)})
        super().__call__(minimal)
        with self._lock:
            for key,row in self._pending.items():
                if key[-1] == capture.request.decision_seq:
                    pending = {k:v for k,v in row.items() if k in RECORD_FIELDS}
                    pending['episode_id'] = pending['library_episode_id']
                    self._pending[key] = pending


class LibraryCaptureFactory:
    """Per-connection sinks, actual-binding shared writers, no model branches."""
    def __init__(self, directory, *, capture_images=True):
        self.directory = Path(directory)
        self.capture_images = capture_images
        self.bridges, self._lock = {}, threading.Lock()

    def __call__(self, core):
        if core.config.enabled or core.config.checkpoints or core.config.recipe.enabled:
            raise ValueError('teacher collection must disable cache/retrieval/recipe')
        if core.sampler.identity != core.spec.default_sampler.identity:
            raise ValueError('teacher binding differs from full native sampler')
        binding = model_binding(core.spec, core.config.adapter_binding['benchmark_protocol'], sampler=core.sampler)
        key = digest(binding)
        with self._lock:
            if key not in self.bridges:
                self.bridges[key] = LibraryCaptureBridge(self.directory / ('binding_'+key[:16]),
                    enabled=True, binding=binding, capture_images=self.capture_images)
        wires = camera_keys(core.spec.feature_spec.real_cameras)
        def metadata(request):
            return dict(core._episode, image_sources={str(i):{'wire_key':wire,
                'camera':core.spec.feature_spec.real_cameras[i], 'cutpoint':'received_wire_uint8'}
                for i,wire in enumerate(wires) if self.capture_images})
        return LibraryDecisionSink(self.bridges[key], metadata, image_keys=wires)

    def close(self):
        return [bridge.close() for bridge in self.bridges.values()]
