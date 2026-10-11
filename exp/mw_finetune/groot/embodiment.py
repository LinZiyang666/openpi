"""Register a named Sawyer tag in-process, without modifying the vendor checkout."""
from __future__ import annotations

from enum import Enum
import sys

from .conventions import EMBODIMENT, EMBODIMENT_ID


def register_embodiment():
    from gr00t.data import embodiment_tags as tags
    current = tags.EmbodimentTag
    if any(member.value == EMBODIMENT for member in current):
        return current(EMBODIMENT)
    members = {member.name: member.value for member in current}
    members["METAWORLD_SAWYER"] = EMBODIMENT
    extended = Enum("EmbodimentTag", members, module=tags.__name__)
    tags.EmbodimentTag = extended
    tags.EMBODIMENT_TAG_MAPPING[EMBODIMENT] = EMBODIMENT_ID
    # Update modules imported by a host before load_policy was called, including
    # Pydantic's enum validator. All subsequent imports receive the same Enum.
    for name, module in list(sys.modules.items()):
        if name.startswith("gr00t.") and getattr(module, "EmbodimentTag", None) is current:
            module.EmbodimentTag = extended
    schema = sys.modules.get("gr00t.data.schema")
    if schema is not None:
        schema.DatasetMetadata.model_fields["embodiment_tag"].annotation = extended
        schema.DatasetMetadata.model_rebuild(force=True)
    return extended(EMBODIMENT)
