"""H100-only episode closure, reusing the teacher allowlist/shared writer."""
from __future__ import annotations

import threading

from .library_capture import LibraryCaptureFactory
from .validate import digest, episode_identity


def _bytes(value):
    return value.numel()*value.element_size() if hasattr(value,'numel') else value.nbytes


class EpisodeShardSink:
    def __init__(self,factory,core):
        self.factory,self.core = factory,core
        self.inner = None
        self.images = None
        self.shard_id = None

    def bind_observation(self,observation):
        from .library_capture import LibraryDecisionSink
        from .attempts import decision_intent
        decision_intent(self.factory.directory, self.core._episode)
        # Same strict validation/copy as the original local path.
        from .library_capture import camera_keys
        holder = LibraryDecisionSink(None,{}, image_keys=camera_keys(self.core.spec.feature_spec.real_cameras),
                                     capture_images=self.factory.capture_images)
        holder.bind_observation(observation)
        self.images = holder._images
        if self.factory.capture_images:
            # Image-free shards are a few MB against a 64 MiB slot; per-decision locked
            # quota rewrites (two fsyncs each) serialized every replica.
            self.factory.budget.check()

    def __call__(self,capture):
        from exp.r17_integration.ops.build_client_bundle import json_bytes
        from exp.r17_integration.ops.sync_back import atomic_bytes
        from .reader_profiles import canonical_episode
        if self.images is None:
            raise ValueError('teacher shard has no bound input images')
        identity = episode_identity(canonical_episode(self.core._episode))
        shard_id = digest(identity)
        if self.inner is None:
            from dataclasses import asdict
            leaf = self.factory.directory/'shards'/shard_id
            self.factory.budget.reserve(shard_id,leaf)
            self.factory.budget.charge(shard_id,4*len(json_bytes(asdict(self.core.spec))))
            atomic_bytes(leaf/'manifest.json',json_bytes(dict(schema='r17.teacher_shard.v1',
                campaign=self.factory.budget.campaign['run_name'],shard_id=shard_id,
                episode_identity=list(identity))))
            self.inner_factory = LibraryCaptureFactory(leaf, capture_images=self.factory.capture_images)
            self.inner = self.inner_factory(self.core)
            self.shard_id = shard_id
            with self.factory.lock:
                self.factory.active[shard_id] = self
        if shard_id!=self.shard_id:
            raise ValueError('episode changed before its terminal capture acknowledgment')
        fields = self.core.spec.feature_spec.fields
        # 2x detached numeric bytes +256KiB per decision covers zip/npy headers,
        # duplicated decision/execution JSON and block metadata conservatively.
        size = sum(_bytes(capture.fields[name]) for name in fields)
        size += _bytes(capture.normalized_chunk)+_bytes(capture.fields['robot_state'])
        size += sum(image.nbytes for image in self.images)
        from .library_capture import RECORD_FIELDS
        from .validate import plain
        metadata = {k:plain(v) for k,v in self.core._episode.items() if k in RECORD_FIELDS}
        if self.factory.capture_images:
            self.factory.budget.charge(shard_id,2*size+2*len(json_bytes(metadata))+256*1024)
        self.inner._images = self.images
        self.inner(capture)

    def on_execution(self,ack):
        if self.inner is None:
            raise ValueError('execution acknowledgment has no active teacher shard')
        if self.factory.capture_images:
            self.factory.budget.check()
        self.inner.on_execution(ack)
        if ack.get('terminal'):
            if self.inner._pending:
                raise ValueError('terminal shard has unacknowledged decisions')
            stats = self.inner_factory.close()
            if any(not row.get('drained') or row.get('errors') for row in stats):
                raise RuntimeError('teacher shard writer failed to drain')
            self.factory.budget.closed(self.shard_id)
            with self.factory.lock:
                del self.factory.active[self.shard_id]
                self.factory.stats.extend(stats)
            self.inner = None
            self.shard_id = None


class ShardedLibraryCaptureFactory:
    def __init__(self,directory,budget, *, capture_images=True):
        from pathlib import Path
        self.directory,self.budget = Path(directory),budget
        self.capture_images = capture_images
        self.active,self.stats,self.lock = {},[],threading.Lock()

    def __call__(self,core):
        return EpisodeShardSink(self,core)

    def close(self):
        # Disconnected/incomplete episodes are retained, never marked streamable.
        with self.lock:
            for sink in self.active.values():
                self.stats.extend(sink.inner_factory.close())
            return list(self.stats)
