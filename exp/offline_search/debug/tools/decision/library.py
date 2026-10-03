"""Read only immutable member actions named by captured library artifacts."""
import hashlib
from pathlib import Path

import numpy as np

from . import common as C


def artifact(arm, lib, filename):
    key = "library_%s_%s" % (lib, filename)
    metas = getattr(arm, "server_metas", {})
    entries = list(metas.values()) if isinstance(metas, dict) else []
    entries.append(getattr(arm, "server_meta", {}))
    found = []
    for meta in entries:
        item = meta.get("artifacts", {}).get(key) if isinstance(meta, dict) else None
        if isinstance(item, dict) and item.get("path") and item.get("status", "available") == "available":
            found.append(item)
    identities = {(item["path"], item.get("sha256")) for item in found}
    if len(identities) > 1:
        raise C.Unavailable("captured processes disagree on library artifact " + key)
    if not found:
        raise C.Unavailable("captured library action artifact path unavailable for " + str(lib))
    path = Path(found[0]["path"])
    if not path.is_file():
        raise C.Unavailable("captured library artifact is absent: " + str(path))
    return path, found[0].get("sha256")


class MemberActions:
    def __init__(self, arm):
        self.arm = arm
        self.banks = {}

    def bank(self, lib):
        if lib not in self.banks:
            path, expected_sha = artifact(self.arm, lib, "action.npy")
            if expected_sha:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected_sha:
                    raise C.Unavailable("library action artifact differs from captured sha256")
            actions = np.load(path, mmap_mode="r", allow_pickle=False)
            if actions.ndim != 3 or actions.dtype.kind not in "fiu":
                raise C.Unavailable("library actions must be numeric row/horizon/channel arrays")
            self.banks[lib] = actions, str(path), expected_sha
        return self.banks[lib]

    def spread(self, row, head):
        """Weighted motion RMS about member mean on the actual applied prefix."""
        lib = C.field(row, "lib")
        if lib is None:
            raise C.Unavailable("live library identifier unavailable")
        actions, path, sha = self.bank(lib)
        rows, weights = C.field(row, "rows"), C.field(row, "weights")
        if rows is None or weights is None:
            raise C.Unavailable("member rows/weights unavailable")
        rows, weights = np.asarray(rows), np.asarray(weights, float)
        if rows.ndim != 1 or weights.shape != rows.shape or not len(rows) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise C.Unavailable("invalid member rows/weights")
        offset = C.number(row.get("chunk_offset"))
        source = C.field(row, "follow_source")
        successor = C.field(row, "successor_rows")
        if row.get("src") == "follow":
            if source == "successor_heads" or C.number(C.field(row, "os_sf_source")) == 2:
                if successor is None:
                    raise C.Unavailable("successor-head member row alignment unavailable")
                rows, offset = np.asarray(successor), 0
            elif source != "native_tail" and C.number(C.field(row, "os_sf_source")) != 1:
                raise C.Unavailable("follow member source alignment unavailable")
        if offset is None or int(offset) != offset or offset < 0 or int(offset) + head > actions.shape[1]:
            raise C.Unavailable("member action prefix is outside the recorded native horizon")
        if rows.shape != weights.shape or rows.dtype.kind not in "iuf" or not np.isfinite(rows).all() or (rows != rows.astype(np.int64)).any():
            raise C.Unavailable("invalid aligned member row identifiers")
        positive = weights > 0
        rows, weights = rows[positive].astype(np.int64), weights[positive] / weights.sum()
        if (rows < 0).any() or (rows >= len(actions)).any():
            raise C.Unavailable("positive-weight member action rows unsupported")
        dims, sigma, grip, _, units = C.dimensions(self.arm, actions.shape[-1])
        motion = [dim for dim in dims if dim != grip]
        if not motion:
            raise C.Unavailable("declared motion action dimensions unavailable")
        chunks = np.asarray(actions[rows, int(offset):int(offset) + head], float)[:, :, motion] / sigma[motion]
        if not np.isfinite(chunks).all():
            raise C.Unavailable("member action prefix is nonfinite")
        mean = np.einsum("n,nhd->hd", weights, chunks)
        variance = np.einsum("n,nhd->hd", weights, (chunks - mean) ** 2)
        return dict(member_spread=float(np.sqrt(variance.mean())), member_spread_status="available",
                    member_spread_reason=None, member_spread_source="weighted immutable member actions",
                    member_spread_offset=int(offset), member_spread_units=units,
                    member_action_path=path, member_action_sha256=sha)
