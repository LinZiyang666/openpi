"""Shared osdebug.v1 identities, sampling and lossless block I/O.

Keep this module usable by the Python 3.8 client; never load pickled arrays.
"""
import hashlib
import json
import logging
import os
from enum import Enum
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "osdebug.v1"
SCHEMA = SCHEMA_VERSION
DECISION_BLOCK_SIZE = 16
CONTROL_BLOCK_SIZE = 64
SERVER_BLOCK_SIZE = DECISION_BLOCK_SIZE
CLIENT_BLOCK_SIZE = CONTROL_BLOCK_SIZE
DECISION_ID_DTYPE = "<U80"
PROMPT_DTYPE = "<U512"
SERVER_BLOCK_PATTERN = r"d_\d+_\d{6}\.npz"
RAWKEY_BLOCK_PATTERN = r"k_\d+_\d{6}\.npz"
CONTROL_BLOCK_PATTERN = r"controls_\d{4}\.npz"
SNAPSHOT_PATTERN = r"snap_(reset|final|\d{6})\.npz"


class Status(str, Enum):
    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"
    NOT_SAMPLED = "not_sampled"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


def status(value, reason=""):
    return {"status": Status(value).value, "reason": str(reason)}


def episode_key(task_uid, attempt):
    attempt = int(attempt)
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    return hashlib.sha256(str(task_uid).encode("utf-8")).hexdigest()[:24] + "_a" + str(attempt)


def decision_id(episode_key, dispatch_gen, decision_seq):
    if int(dispatch_gen) < 0 or int(decision_seq) < 0:
        raise ValueError("generation and sequence must be nonnegative")
    result = "{}:{}:{}".format(episode_key, int(dispatch_gen), int(decision_seq))
    if len(result) > 80:
        raise ValueError("decision_id exceeds schema width")
    return result


def sampling_hash(campaign, task_uid, decision_seq):
    payload = "{}|{}|{}".format(campaign, task_uid, int(decision_seq))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")


def rawkeys_sampled(campaign, task_uid, decision_seq):
    return sampling_hash(campaign, task_uid, decision_seq) % 16 == 0


def snapshot_sampled(campaign, task_uid, decision_seq):
    return int(decision_seq) <= 0 or sampling_hash(campaign, task_uid, decision_seq) % 16 == 1


def draws_sampled(campaign, task_uid, decision_seq):
    return sampling_hash(campaign, task_uid, decision_seq) % 32 == 2


def _safe_arrays(arrays):
    result = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("object array forbidden: " + str(key))
        if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and not np.little_endian):
            array = array.astype(array.dtype.newbyteorder("<"))
        result[str(key)] = array
    return result


def write_npz_block(path, arrays):
    """Atomically publish a compressed block; a failure retains its .part."""
    path = Path(path)
    arrays = _safe_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(part), str(path))
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_npz_block(path):
    with np.load(str(path), allow_pickle=False) as data:
        return _safe_arrays({key: data[key].copy() for key in data.files})


def read_jsonl(path, issues=None):
    """Yield complete JSONL rows; report and skip an unterminated final line.

    A newline is the publish boundary even when the final bytes are valid JSON.
    Corruption in a completed line remains an error, rather than silently losing
    a record. Binary reads also tolerate a killed process's partial UTF-8 code.
    """
    with Path(path).open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                issue = dict(path=str(path), line=line_number, bytes=len(line),
                             reason="unterminated final JSONL line skipped")
                if issues is not None:
                    issues.append(issue)
                logging.getLogger("osdebug").warning("%s:%s: %s", path, line_number, issue["reason"])
                continue
            if line.strip():
                yield json.loads(line.decode("utf-8"))
