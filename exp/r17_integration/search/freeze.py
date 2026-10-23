"""Seal every numerical and cohort dependency before formal test admission."""
from __future__ import annotations

from exp.r17_integration.debug.validate import digest, file_sha, plain
from .store import atomic_json

DEPENDENCIES = ("yaml", "model", "norm", "layout", "sampler", "fit", "library", "protocol", "split", "code")


def freeze(dependencies, path=None):
    if set(dependencies) - {"code"} != set(DEPENDENCIES) - {"code"}:
        raise ValueError("freeze requires all dependency identities")
    value = dict(version="r17.freeze.v1", dependencies=plain(dependencies))
    value["digest"] = digest(value)
    if path is not None:
        atomic_json(path, value)
    return value


def file_dependencies(paths):
    return {name: file_sha(path) for name, path in paths.items()}
