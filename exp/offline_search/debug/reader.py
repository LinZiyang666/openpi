"""Accepted-attempt reader for osdebug.v1 and a read-only R6 P3 v2 adapter.

Arrays are always joined by decision_id, never by file arrival or decision
number across arms. Missing arrays raise KeyError rather than fabricate zeros.
``accepted_only=False`` retains retries and error prefixes for audits.
"""
import hashlib
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .fixtures import _episode_key


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def iter_jsonl(path, issues=None):
    """Yield (record, byte count); an unpublished final line is skipped/reported."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("rb") as f:
        for i, line in enumerate(f, 1):
            if not line.endswith(b"\n"):
                issue = dict(code="truncated_jsonl_tail", where=str(path), line=i, bytes=len(line),
                             detail="final line without newline skipped")
                if issues is not None:
                    if issue not in issues:
                        issues.append(issue)
                else:
                    warnings.warn("%s:%d: final line without newline skipped" % (path, i), RuntimeWarning)
                continue
            if line.strip():
                try:
                    yield json.loads(line), len(line)
                except (ValueError, TypeError, UnicodeError) as exc:
                    raise ValueError("%s:%d: invalid JSON: %s" % (path, i, exc)) from exc


def read_jsonl(path, issues=None):
    return [row for row, _ in iter_jsonl(path, issues)]


def read_npz(path, keys=None):
    """NPZ is lazy: requesting small fields never inflates unrelated images."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else _list(keys))}


def uid_identity(row):
    """Subset init is authoritative; preserve older logged original init indices."""
    row = dict(row)
    parts = str(row.get("task_uid", "")).rsplit(":", 2)
    if len(parts) == 3:
        task, init = int(parts[-2]), int(parts[-1])
        if row.get("init") is not None and int(row["init"]) != init:
            row.setdefault("orig_init_state_idx", int(row["init"]))
        row.update(task_id=task, init=init)
    return row


def _list(value):
    return [value] if isinstance(value, str) else list(value)


def _unique(frame, keys, label):
    if not frame.empty and frame.duplicated(keys).any():
        raise ValueError("duplicate %s: %s" % (label, frame.loc[frame.duplicated(keys, keep=False), keys].to_dict("records")))


def _arm_path(root, name):
    root = Path(root)
    if (root / "debug").is_dir() or (root / "client/journal.jsonl").exists():
        return root
    return root / "runs" / name


class ArmData:
    def __init__(self, run_root, arm_name):
        self.arm_name = arm_name
        self.arm_dir = _arm_path(run_root, arm_name)
        self.run_root = self.arm_dir.parent.parent if self.arm_dir.parent.name == "runs" else Path(run_root)
        self.debug_dir = self.arm_dir / "debug"
        self.manifest = read_json(self.debug_dir / "MANIFEST.json")
        spec = self.manifest.get("arm_spec", {})
        round_spec = spec.get("r8", {})
        for field in ("model", "suite", "method"):
            if field in spec:
                self.manifest.setdefault(field, spec[field])
        if round_spec.get("augmentation"):
            self.manifest.setdefault("required_aug", round_spec["augmentation"])
        if "library" in round_spec:
            self.manifest.setdefault("lib", round_spec["library"])
        if round_spec.get("variant") == "P10":
            self.manifest.setdefault("pure_policy", True)
        self.server_dirs = sorted(p for p in self.debug_dir.glob("server_*") if p.is_dir())
        self.server_metas = {str(p.relative_to(self.debug_dir)): read_json(p)
                             for d in self.server_dirs for p in sorted(d.glob("meta*.json"))}
        self.server_meta = next(iter(self.server_metas.values()), {})
        self._indexes = {}
        self.read_issues = []
        self.cache_enabled = True

    def _jsonl(self, path):
        return read_jsonl(path, self.read_issues)

    def record_meta(self, row):
        directory = Path(row["_server_dir"])
        pid = row.get("pid")
        if pid is not None:
            pid = None if pd.isna(pid) else int(pid)
        if pid is None:
            filename = Path(row.get("_server_file", "")).stem
            if filename.startswith("decisions_"):
                pid = filename[len("decisions_"):]
        if pid is not None:
            path = directory / ("meta_%s.json" % pid)
            key = str(path.relative_to(self.debug_dir))
            if key not in self.server_metas and path.exists():
                self.server_metas[key] = read_json(path)
            if key in self.server_metas:
                return self.server_metas[key]
        old = str((directory / "meta.json").relative_to(self.debug_dir))
        if old in self.server_metas and (Path(row.get("_server_file", "")).name == "decisions.jsonl" or
                                        self.server_metas[old].get("pid") == pid):
            return self.server_metas[old]
        candidates = [v for k, v in self.server_metas.items() if Path(k).parent.name == directory.name]
        return candidates[0] if pid is None and len(candidates) == 1 else {}

    def _cached(self, name, sources, build):
        """Disposable parquet cache with input stat fingerprint and JSON columns."""
        if not self.cache_enabled:
            return build()
        signature = hashlib.sha256(json.dumps(["reader.fix1.2", [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                                                for p in sources if p.exists()]]).encode()).hexdigest()
        target = self.debug_dir / "derived" / (name + ".parquet")
        meta_path = target.with_suffix(".json")
        metadata = read_json(meta_path)
        if metadata.get("signature") == signature and target.exists():
            try:
                df = pd.read_parquet(target)
                for issue in metadata.get("read_issues", []):
                    if issue not in self.read_issues:
                        self.read_issues.append(issue)
                for col in metadata.get("json_columns", []):
                    df[col] = df[col].map(lambda v: json.loads(v) if isinstance(v, str) else v)
                return df
            except (ValueError, OSError):
                pass
        df = build()
        # Historical captures must remain read-only, even if opened accidentally
        # through this reader. In-memory results remain usable without a cache.
        if str(self.run_root.resolve()).startswith("/home/weiland/trace_runs"):
            return df
        encoded = df.copy()
        columns = []
        for col in encoded:
            if encoded[col].map(lambda v: isinstance(v, (dict, list))).any():
                columns.append(col)
                encoded[col] = encoded[col].map(lambda v: json.dumps(v, default=str))
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            encoded.to_parquet(target.with_suffix(".part"), index=False)
            os.replace(str(target.with_suffix(".part")), str(target))
            meta_path.write_text(json.dumps(dict(signature=signature, json_columns=columns,
                                                read_issues=[i for i in self.read_issues if i["where"] in {str(p) for p in sources}])))
        except (OSError, ValueError, ImportError):
            pass
        return df

    @property
    def journal_path(self):
        for path in (self.arm_dir / "client/journal.jsonl", self.arm_dir / "journal.jsonl"):
            if path.exists():
                return path
        return self.arm_dir / "client/journal.jsonl"

    def journal(self):
        """One authoritative row per attempt; terminal journal rows supersede starts."""
        def build():
            attempts = {}
            for row in self._jsonl(self.journal_path):
                if "task_uid" not in row:
                    continue
                row = uid_identity(row)
                row["attempt"] = int(row.get("attempt", 1))
                key = row["task_uid"], row["attempt"]
                old = attempts.get(key, {})
                if old.get("accepted") and row.get("accepted") and any(
                        old.get(k) != row.get(k) for k in ("success", "status", "error") if k in old and k in row):
                    raise ValueError("conflicting accepted journal outcome: %s" % (key,))
                attempts[key] = dict(old, **row)
            result = []
            for row in attempts.values():
                row.setdefault("accepted", False)
                row.setdefault("success", None)
                parts = row["task_uid"].split(":")
                if len(parts) >= 4:
                    row.setdefault("task_id", int(parts[-2]))
                    row.setdefault("init", int(parts[-1]))
                row["episode_key"] = _episode_key(row["task_uid"], row["attempt"])
                result.append(row)
            df = pd.DataFrame(result)
            for c in ("task_uid", "task_id", "init", "attempt", "accepted", "success", "episode_key"):
                if c not in df:
                    df[c] = pd.Series(dtype=object)
            accepted = df[df.accepted.eq(True)]
            _unique(accepted, ["task_uid"], "accepted task_uid")
            _unique(accepted, ["task_id", "init"], "accepted task/init")
            return df
        return self._cached("journal", [self.journal_path], build)

    def _outcomes(self, frame, accepted_only):
        journal = self.journal()
        cols = [c for c in ("episode_key", "accepted", "success", "status", "error") if c in journal]
        outcomes = journal[cols].rename(columns={"success": "journal_success", "status": "journal_status", "error": "journal_error"})
        out = frame.merge(outcomes, on="episode_key", how="left", validate="many_to_one")
        out["accepted"] = out.accepted.eq(True)
        return out[out.accepted].reset_index(drop=True) if accepted_only else out

    def episodes(self, accepted_only=True):
        paths = sorted((self.debug_dir / "client").glob("*/episode.json"))
        def build():
            rows = []
            for path in paths:
                row = uid_identity(read_json(path))
                row.setdefault("episode_key", path.parent.name)
                rows.append(row)
            df = pd.DataFrame(rows)
            if "episode_key" not in df:
                df["episode_key"] = pd.Series(dtype=str)
            _unique(df, ["episode_key"], "client episode")
            return df
        return self._outcomes(self._cached("episodes", paths, build), accepted_only)

    def server_records(self):
        rows = []
        for directory in self.server_dirs:
            for path in sorted(directory.glob("decisions*.jsonl")):
                for row in self._jsonl(path):
                    if "decision_id" in row:
                        row = dict(row)
                        if path.name == "decisions.jsonl" and "orig_init_state_idx" not in row and row.get("init") is not None:
                            # In the old single-file writer, init was explicitly
                            # the original state index, even when equal to UID init.
                            row["orig_init_state_idx"] = int(row["init"])
                        rows.append(dict(uid_identity(row), _server_dir=str(directory), _server_file=str(path)))
        return pd.DataFrame(rows, columns=None if rows else ["decision_id", "episode_key", "decision_seq"])

    def client_events(self):
        rows = []
        for path in sorted((self.debug_dir / "client").glob("*/events.jsonl")):
            for row in self._jsonl(path):
                if "decision_id" in row:
                    rows.append(dict(row, episode_key=path.parent.name))
        return pd.DataFrame(rows, columns=None if rows else ["decision_id", "episode_key"])

    def decisions(self, columns=None, accepted_only=True):
        sources = sorted(p for d in self.server_dirs for p in d.glob("decisions*.jsonl")) + sorted((self.debug_dir / "client").glob("*/events.jsonl"))
        def build():
            server, client = self.server_records(), self.client_events()
            _unique(server, ["decision_id"], "server decision_id")
            _unique(client, ["decision_id"], "client decision_id")
            out = server.merge(client, how="outer", on="decision_id", suffixes=("", "_client"), indicator="_join", validate="one_to_one")
            if "episode_key_client" in out:
                conflict = out.episode_key.notna() & out.episode_key_client.notna() & out.episode_key.ne(out.episode_key_client)
                if conflict.any():
                    raise ValueError("server/client episode identity conflict")
                out["episode_key"] = out.episode_key.fillna(out.episode_key_client)
            if "decision_seq_client" in out:
                conflict = out.decision_seq.notna() & out.decision_seq_client.notna() & out.decision_seq.ne(out.decision_seq_client)
                if conflict.any():
                    raise ValueError("server/client decision sequence conflict")
                out["decision_seq"] = out.decision_seq.fillna(out.decision_seq_client)
            out["server_join"] = out["_join"].map({"both": "verified", "left_only": "missing_client", "right_only": "missing_server"}).astype(str)
            return out.drop(columns="_join")
        out = self._outcomes(self._cached("decisions", sources, build), accepted_only)
        if "decision_seq" in out:
            out = out.sort_values(["episode_key", "decision_seq"], kind="stable").reset_index(drop=True)
        return out if columns is None else out[_list(columns)]

    def _array_index(self, family, paths):
        signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
        if family in self._indexes and self._indexes[family][0] == signature:
            return self._indexes[family][1]
        index = {}
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                if "decision_id" not in z:
                    raise ValueError("missing decision_id: %s" % path)
                for i, did in enumerate(z["decision_id"].astype(str)):
                    if did in index:
                        raise ValueError("duplicate array decision_id %s in %s" % (did, path))
                    index[did] = path, i
        self._indexes[family] = signature, index
        return index

    def _aligned_arrays(self, family, paths, keys, decision_ids):
        index = self._array_index(family, paths)
        ids = list(index) if decision_ids is None else _list(decision_ids)
        missing = [did for did in ids if did not in index]
        if missing:
            raise KeyError("%s missing decision_ids: %s" % (family, missing))
        if not ids:
            if paths:
                actual_keys = None if keys is None else [k for k in _list(keys) if k != "prompt"] + ["decision_id"]
                if keys is not None and "prompt" in _list(keys):
                    actual_keys += ["prompts", "prompt_idx"]
                sample = read_npz(paths[0], actual_keys)
                selected = [k for k, v in sample.items() if v.ndim and len(v) == len(sample["decision_id"])] if keys is None else _list(keys)
                return {k: (sample["prompts"][sample["prompt_idx"]] if k == "prompt" else sample[k])[:0] for k in selected}
            return {k: np.array([], dtype=str if k == "decision_id" else float) for k in ([] if keys is None else _list(keys))}
        groups = {}
        for output_idx, did in enumerate(ids):
            path, row_idx = index[did]
            groups.setdefault(path, []).append((output_idx, row_idx))
        result = {}
        for path, selected_rows in groups.items():
            with np.load(path, allow_pickle=False) as archive:
                members = list(archive.files) if keys is None else list(dict.fromkeys(
                    [k for k in _list(keys) if k != "prompt"] + ["decision_id"] +
                    (["prompts", "prompt_idx"] if "prompt" in _list(keys) else [])))
                for k in list(members):
                    if k.startswith("img_") and not k.endswith("_available") and k + "_available" in archive.files:
                        members.append(k + "_available")
                missing_fields = set(members) - set(archive.files)
                if missing_fields:
                    raise KeyError("%s missing field %s in %s" % (family, sorted(missing_fields), path))
                data = {k: archive[k] for k in dict.fromkeys(members)}
            selected_keys = [k for k, v in data.items() if v.ndim and len(v) == len(data["decision_id"]) and k != "prompts"] if keys is None else _list(keys)
            if keys is None and result and set(selected_keys) != set(result):
                raise KeyError("inconsistent fields across %s parts" % family)
            for k in selected_keys:
                if k == "prompt":
                    value = data["prompts"][data["prompt_idx"]]
                else:
                    if k not in data:
                        raise KeyError("%s missing field %s in %s" % (family, k, path))
                    value = data[k]
                mask = data.get(k + "_available") if k.startswith("img_") and not k.endswith("_available") else None
                if mask is not None:
                    if mask.dtype != np.bool_ or mask.shape != data["decision_id"].shape:
                        raise ValueError("invalid image availability mask in %s" % path)
                    for _, row_idx in selected_rows:
                        if not mask[row_idx]:
                            raise KeyError("wire image %s unavailable for %s" % (k, data["decision_id"][row_idx]))
                if not value.ndim or len(value) != len(data["decision_id"]):
                    raise ValueError("%s is not decision aligned in %s" % (k, path))
                if k not in result:
                    result[k] = np.empty((len(ids),) + value.shape[1:], dtype=value.dtype)
                elif result[k].shape[1:] != value.shape[1:] or result[k].dtype != value.dtype:
                    # Unicode widths may vary between part files.
                    if result[k].shape[1:] == value.shape[1:] and result[k].dtype.kind in "US" and value.dtype.kind in "US":
                        result[k] = result[k].astype(np.promote_types(result[k].dtype, value.dtype))
                    else:
                        raise ValueError("shape/dtype conflict for %s in %s" % (k, family))
                for output_idx, row_idx in selected_rows:
                    result[k][output_idx] = value[row_idx]
        return result

    def decision_arrays(self, keys, decision_ids):
        keys = _list(keys)
        raw = [k for k in keys if k.startswith("raw_key_")]
        regular = [k for k in keys if k not in raw]
        result = {}
        for family, selected in (("blocks", regular), ("rawkeys", raw)):
            if selected:
                paths = sorted(p for d in self.server_dirs for p in (d / family).glob("*.npz"))
                result.update(self._aligned_arrays(family, paths, selected, decision_ids))
        return result

    def controls(self, episode_key, keys=None):
        paths = sorted((self.debug_dir / "client" / episode_key).glob("controls_*.npz"))
        if not paths:
            raise KeyError("missing controls for %s" % episode_key)
        selected = None if keys is None else _list(keys)
        chunks, offset, field_set = {}, 0, None
        for path in paths:
            data = read_npz(path, selected)
            if field_set is None:
                field_set = set(data)
            elif selected is None and set(data) != field_set:
                raise ValueError("control fields differ between blocks: %s" % path)
            for k in (list(data) if selected is None else selected):
                if k not in data:
                    raise KeyError("missing control field %s in %s" % (k, path))
                value = data[k]
                if k == "contact_off":
                    value = value + offset
                    if chunks.get(k):
                        value = value[1:]
                chunks.setdefault(k, []).append(value)
            if "contact_off" in data:
                offset += int(data["contact_off"][-1])
        result = {}
        for k, values in chunks.items():
            if k == "act_substep" and len({v.shape[1] for v in values}) > 1:
                width = max(v.shape[1] for v in values)
                values = [np.pad(v, ((0, 0), (0, width - v.shape[1]), (0, 0)), constant_values=np.nan) for v in values]
            result[k] = np.concatenate(values, axis=0)
        return result

    def iter_controls(self, keys=None, accepted_only=True):
        for ek in self.episodes(accepted_only=accepted_only).episode_key:
            yield ek, self.controls(ek, keys)

    def images(self, decision_ids, cameras=None):
        if cameras is None:
            cameras = self.server_meta.get("camera_names", self.server_meta.get("cameras", []))
            if isinstance(cameras, dict):
                cameras = list(cameras)
            cameras = [v.get("name") if isinstance(v, dict) else v for v in cameras]
            if not cameras:
                cameras = list(self.server_meta.get("image_shapes", {}))
        cameras = _list(cameras)
        data = self.decision_arrays(["img_" + c for c in cameras], decision_ids)
        return {c: data["img_" + c] for c in cameras}

    def snapshots(self, episode_key):
        return {p.stem[len("snap_"):]: read_npz(p) for p in sorted((self.debug_dir / "client" / episode_key).glob("snap_*.npz"))}

    def aug(self, kind, decision_ids=None):
        paths = sorted((self.debug_dir / "aug" / kind).glob("part_*.npz"))
        return self._aligned_arrays("aug_" + kind, paths, None, decision_ids)

    def catalog(self):
        from .catalog import catalog_key
        lib = self.manifest.get("lib", self.manifest.get("library", "current")) or "current"
        model = self.manifest.get("model") or self.server_meta.get("model")
        suite = self.manifest.get("suite") or self.server_meta.get("suite")
        key = catalog_key(model, suite, lib)
        reference = self.manifest.get("catalog", self.manifest.get("catalog_path"))
        paths = []
        if reference:
            p = Path(reference)
            p = p if p.is_absolute() else self.run_root / p
            p = p if p.suffix == ".parquet" else p / "rows.parquet"
            if p.parent.name != key:
                raise ValueError("catalog reference must be keyed by model/suite/library: " + key)
            paths.append(p)
        paths.append(self.run_root / "catalog" / key / "rows.parquet")
        for path in paths:
            if path.exists():
                return pd.read_parquet(path)
        return None


def open_arm(run_root, arm_name):
    return ArmData(run_root, arm_name)


def pair(arm_a, arm_b):
    a, b = arm_a.episodes(), arm_b.episodes()
    for data in (a, b):
        _unique(data, ["task_id", "init"], "paired accepted task/init")
    return a.merge(b, on=["task_id", "init"], how="inner", suffixes=("_a", "_b"), validate="one_to_one")


class P3V2ArmData(ArmData):
    """Development view; missing R8 fields remain absent, never guessed.

    P3 raw geom IDs have a known naming defect. ``capabilities`` and adapter
    manifest explicitly mark contacts as unverified. Images are unavailable.
    """
    def __init__(self, run_root, arm_name):
        super().__init__(run_root, arm_name)
        self.manifest = read_json(self.arm_dir / "manifest.json")
        self.manifest.update(adapter="r6p3.v2", schema="r6p3.v2", images_status="unsupported",
                             contacts_status="unverified_legacy_geom_names")
        self.server_dirs = sorted(p for p in self.arm_dir.glob("server_*") if p.is_dir())
        self._p3_rows = None
        self._p3_archive = {}
        self._p3_telemetry = {}
        self._p3_tables = {}
        self._p3_catalog = {}

    def _cached(self, name, sources, build):
        # The development adapter never writes to legacy trees, including
        # synthetic P3 copies outside the protected trace_runs root.
        return build()

    def _table(self, name):
        if name in self._p3_tables:
            return self._p3_tables[name]
        frames = []
        for path in sorted((self.run_root / "tables").glob("*/" + name + ".csv")):
            model = self._p3_catalog.get("model")
            if model and not path.parent.name.startswith(model + "_"):
                continue
            if path.stat().st_size == 0:
                continue
            for frame in pd.read_csv(path, chunksize=20000, low_memory=False):
                if "arm" in frame:
                    selected = frame.loc[frame.arm.eq(self.arm_name)]
                    if not selected.empty:
                        frames.append(selected)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self._p3_tables[name] = result
        return result

    def _load(self):
        if self._p3_rows is not None:
            return
        records, startup = {}, {}
        for directory in self.server_dirs:
            for path in sorted(directory.glob("decisions_*.jsonl")):
                for row in read_jsonl(path):
                    if row.get("ev") == "p3_v2_startup":
                        startup = row
                    if row.get("ev") not in ("dec", "p3_anchor", "p3_decision"):
                        continue
                    uid = row.get("uid", row.get("task_uid"))
                    key = uid, int(row.get("attempt", 1) or 1), int(row["step"])
                    records.setdefault(key, {})[row["ev"]] = dict(row, _server_dir=str(directory))
        result = []
        for (uid, attempt, seq), group in records.items():
            base, detail, anchor = group.get("dec", {}), group.get("p3_decision", {}), group.get("p3_anchor", {})
            row = dict(base, **detail)
            ek = _episode_key(uid, attempt)
            did = "%s:0:%d" % (ek, seq)
            parts = uid.split(":")
            row.update(task_uid=uid, episode_key=ek, attempt=attempt, dispatch_gen=0, decision_seq=seq, decision_id=did,
                       task_id=int(parts[-2]), init=int(parts[-1]), camera_mode="full" if row.get("vision") else "blind",
                       blind_age_controls=0 if row.get("vision") else 5, diag=anchor or detail,
                       source_schema="r6p3.v2")
            # Deployed counts exclude P3's profiling-only forwards.
            row.update(stage1_calls=int(bool(row.get("vision"))), policy_calls=int(not bool(row.get("hit", True))),
                       stage23_calls=int(not bool(row.get("hit", True))), camera_completions=0)
            if anchor:
                retrieval = anchor.get("retrieval", {})
                for k in ("rows", "weights", "scores", "d1", "conf"):
                    if k in retrieval:
                        row[k] = retrieval[k]
                assign = anchor.get("assignment", {})
                row.update(p_nominal=assign.get("propensity"), p_effective=assign.get("actual_propensity"),
                           coin=assign.get("coin"), eligible=assign.get("eligible"),
                           treatment=assign.get("executed_policy"), override=assign.get("override"))
            if detail.get("input_archive"):
                self._p3_archive[did] = Path(detail["_server_dir"]) / detail["input_archive"]
            result.append(row)
        self._p3_catalog = startup.get("catalog", {})
        table = self._table("decisions")
        if not table.empty:
            table = table.copy()
            table["task_uid"] = table.uid
            table["episode_key"] = [_episode_key(uid, int(attempt)) for uid, attempt in zip(table.uid, table.attempt)]
            table["decision_seq"] = table.step.astype(int)
            table["decision_id"] = ["%s:0:%d" % (ek, seq) for ek, seq in zip(table.episode_key, table.decision_seq)]
            _unique(table, ["decision_id"], "P3 table decision_id")
            for _, row in table.iterrows():
                path = row.get("absolute_input_archive")
                if isinstance(path, str):
                    self._p3_archive.setdefault(row.decision_id, Path(path))
            if result:
                raw = pd.DataFrame(result)
                # Keep authoritative raw fields; tables add checked application
                # and flattened diagnostic columns without replacing records.
                extra = [c for c in table if c not in raw or c == "decision_id"]
                self._p3_rows = raw.merge(table[extra], how="left", on="decision_id", validate="one_to_one")
            else:
                self._p3_rows = table
                if "src" not in table and "source" in table:
                    self._p3_rows["src"] = table.source
                self._p3_rows["stage1_calls"] = table.vision.astype(int)
                self._p3_rows["policy_calls"] = (~table.hit.astype(bool)).astype(int) if "hit" in table else np.nan
                self._p3_rows["stage23_calls"] = self._p3_rows.policy_calls
                self._p3_rows["camera_completions"] = 0
        else:
            self._p3_rows = pd.DataFrame(result, columns=None if result else ["decision_id", "episode_key", "decision_seq"])
        catalog = startup.get("catalog", {})
        interface = catalog.get("interface", catalog.get("manifest", {}))
        model = catalog.get("model", next((r.get("model") for r in result if r.get("model")), "pi05"))
        if not result and "model" in self._p3_rows and not self._p3_rows.empty:
            model = self._p3_rows.model.iloc[0]
        self.server_meta = dict(interface, model=model, H=10 if model == "pi05" else 16, action_dim=32,
                                state_dim=8, valid_action_dims=list(range(7)), camera_names=["third", "wrist"])
        for path in sorted((self.arm_dir / "client_telemetry").glob("*/controls.jsonl")):
            self._p3_telemetry[path.parent.name] = path

    def episodes(self, accepted_only=True):
        self._load()
        rows = []
        for journal in self.journal().to_dict("records"):
            telemetry = read_jsonl(self._p3_telemetry[journal["episode_key"]]) if journal["episode_key"] in self._p3_telemetry else []
            start = next((r for r in telemetry if r.get("ev") == "rollout_start"), {})
            end = next((r for r in reversed(telemetry) if r.get("ev") == "rollout_end"), {})
            row = dict(journal, **{k: v for k, v in start.items() if k not in journal})
            row.update(n_controls=end.get("controls"), n_decisions=end.get("decisions"),
                       env_seed=end.get("environment_seed", start.get("environment_seed", telemetry[0].get("environment_seed") if telemetry else None)),
                       termination_reason=end.get("termination_reason"),
                       entities=next((r.get("after", {}).get("entity_ids") for r in telemetry if r.get("ev") == "control"), {}),
                       capabilities=dict(contacts=dict(status="unsupported", reason="legacy geom naming is unverified")),
                       physical_transitions_available=bool(telemetry), source_schema="r6p3.v2")
            row["journal_success"] = journal.get("success")
            rows.append(row)
        frame = pd.DataFrame(rows)
        table = self._table("episodes")
        if not table.empty and not frame.empty:
            table = table.copy()
            table["episode_key"] = [_episode_key(uid, int(attempt)) for uid, attempt in zip(table.uid, table.attempt)]
            _unique(table, ["episode_key"], "P3 episode table")
            extra = [c for c in table if c not in frame or c == "episode_key"]
            frame = frame.merge(table[extra], how="left", on="episode_key", validate="one_to_one")
        return frame[frame.accepted.eq(True)].reset_index(drop=True) if accepted_only else frame

    def decisions(self, columns=None, accepted_only=True):
        self._load()
        rows = self._outcomes(self._p3_rows.copy(), accepted_only)
        for ek, path in self._p3_telemetry.items():
            events = {int(r["decision_step"]): r for r in read_jsonl(path) if r.get("ev") == "decision"}
            controls = [r for r in read_jsonl(path) if r.get("ev") == "control"]
            for seq, event in events.items():
                mask = rows.episode_key.eq(ek) & rows.decision_seq.eq(seq)
                applied = [r for r in controls if r.get("decision_step") == seq]
                rows.loc[mask, "control_idx_start"] = applied[0]["control"] if applied else np.nan
                rows.loc[mask, "n_applied"] = len(applied)
                rows.loc[mask, "server_join"] = "legacy_unverified"
        return rows.reset_index(drop=True) if columns is None else rows[_list(columns)].reset_index(drop=True)

    def decision_arrays(self, keys, decision_ids):
        self._load()
        aliases = dict(state_wire="raw_state", state_norm="robot_state", served_chunk="executed_chunk",
                       served_wire="wire_chunk", cache_chunk="diagnostic_cache_chunk", raw_key_third="vision_0", raw_key_wrist="vision_1")
        values = {k: [] for k in _list(keys)}
        for did in _list(decision_ids):
            if did not in self._p3_archive:
                raise KeyError("P3 input unavailable for %s" % did)
            data = read_npz(self._p3_archive[did], [aliases.get(k, k) for k in values])
            for k in values:
                source = aliases.get(k, k)
                if source not in data:
                    raise KeyError("P3 %s unavailable for %s" % (k, did))
                values[k].append(data[source])
        return {k: np.stack(v) if v else np.array([]) for k, v in values.items()}

    def controls(self, episode_key, keys=None):
        self._load()
        if episode_key not in self._p3_telemetry:
            raise KeyError("P3 controls unavailable for %s" % episode_key)
        rows = [r for r in read_jsonl(self._p3_telemetry[episode_key]) if r.get("ev") == "control"]
        result = dict(control_idx=np.array([r["control"] for r in rows], np.int32),
                      decision_seq=np.array([r.get("decision_step") if r.get("decision_step") is not None else -1 for r in rows], np.int32),
                      chunk_offset=np.array([r.get("chunk_offset") if r.get("chunk_offset") is not None else -1 for r in rows], np.int32),
                      is_settle=np.array([r.get("decision_step") is None for r in rows]),
                      action=np.array([r["action_issued"] for r in rows], np.float64),
                      reward=np.array([r["reward"] for r in rows], np.float64), done=np.array([r["done"] for r in rows], bool))
        # P3's robot/object representation is nested; preserve numeric arrays
        # where present and retain the full record for physical-tool adapters.
        result["legacy_records"] = rows
        for key in ("qpos", "qvel", "eef_pos", "eef_quat", "gripper_qpos", "gripper_qvel"):
            values = []
            for row in rows:
                after = row.get("after", {})
                source = after.get("robot", {}).get(key, after.get("sim", {}).get(key, after.get(key)))
                if source is None:
                    source = after.get("observation_numeric", {}).get("robot0_" + key)
                if source is None:
                    break
                values.append(source)
            if len(values) == len(rows):
                result[key] = np.asarray(values, np.float64)
        if rows:
            numeric = rows[0].get("after", {}).get("observation_numeric", {})
            objects = sorted(k[:-4] for k in numeric if k.endswith("_pos") and not k.startswith("robot") and "_to_" not in k)
            if objects:
                for key, suffix in (("obj_pos", "_pos"), ("obj_quat", "_quat")):
                    if all(all(name + suffix in r.get("after", {}).get("observation_numeric", {}) for name in objects) for r in rows):
                        result[key] = np.asarray([[r["after"]["observation_numeric"][name + suffix] for name in objects] for r in rows], np.float64)
                result["legacy_object_names"] = objects
            predicates = sorted(rows[0].get("after", {}).get("predicates", {}))
            if predicates:
                result["legacy_predicate_names"] = predicates
                result["predicates"] = np.asarray([[r.get("after", {}).get("predicates", {}).get(name, np.nan) for name in predicates] for r in rows], np.float64)
            if all(r.get("actuator_ctrl_each_physics_step") is not None for r in rows):
                result["act_substep"] = np.asarray([r["actuator_ctrl_each_physics_step"] for r in rows], np.float32)
        return result if keys is None else {k: result[k] for k in _list(keys)}

    def images(self, decision_ids, cameras=None):
        raise KeyError("P3 v2 did not retain wire images; images are unsupported")

    def snapshots(self, episode_key):
        self._load()
        path = self._p3_telemetry.get(episode_key)
        return {} if path is None else {p.stem: read_npz(p) for p in sorted(path.parent.rglob("step_*.npz"))}

    def aug(self, kind, decision_ids=None):
        self._load()
        if kind != "policy_shadow":
            raise KeyError("P3 augmentation %s unavailable" % kind)
        ids = list(self._p3_archive) if decision_ids is None else _list(decision_ids)
        data = self.decision_arrays(["policy_chunk"], ids)
        return dict(decision_id=np.array(ids), chunk=data["policy_chunk"], origin=np.array(["legacy_inline"] * len(ids)))


def p3v2_adapter(run_root, arm_name):
    arm = P3V2ArmData(run_root, arm_name)
    arm._load()
    return arm
