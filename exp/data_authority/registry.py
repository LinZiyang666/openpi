"""Load, validate and query the authoritative dataset ledger.

This module owns the record format under ``records/``. One JSON file describes
one dataset: where its canonical copy lives, what it contains, how it was
produced, and what is known to be odd about it. The bytes themselves are not
stored here -- experiment corpora run to tens of gigabytes and live on the node
that produced them (see ``authority.node``).

It also owns the ``analysis/<task>/MANIFEST.json`` format: collected figures
and reports are registered the same way datasets are, by digest and source, so
that nothing under ``analysis/`` is an unattributable orphan.

Public interface: ``load_record``, ``load_all``, ``validate_record``, ``find``,
``record_path_for``, ``load_analysis_manifest``, ``load_all_analysis``,
``validate_analysis_manifest``, ``analysis_tasks``. The module is also a CLI
(``ls`` / ``show`` / ``validate`` / ``analysis``).

Depends only on the standard library so it stays runnable on any node.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Format constants
# ------------------------------------------------------------------

SCHEMA_VERSION = 1
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
RECORDS_DIR = _HERE / "records"
ANALYSIS_DIR = _HERE / "analysis"
MANIFEST_NAME = "MANIFEST.json"

#: dataset_id path separator, replaced by this token in the filename. Records
#: are one-per-file so that a diff over the ledger reads as a list of changed
#: datasets rather than one churning blob.
ID_SEP = "/"
FILE_SEP = "__"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9_]+(?:/[a-z0-9_.]+)+$")

REQUIRED_TOP = (
    "schema_version",
    "dataset_id",
    "kind",
    "title",
    "experiment",
    "status",
    "authority",
    "integrity",
    "content",
    "provenance",
)

KNOWN_KINDS = (
    "cache_artifact",
    "collection_h5",
    "journal",
    "checkpoint",
    "init_pool",
    #: Measured numbers rather than corpus: benchmark result sets whose
    #: bytes are small but whose provenance (host, build flags, what was
    #: superseded) is exactly what a later reader needs.
    "benchmark_results",
)
KNOWN_STATUS = ("authoritative", "superseded")
KNOWN_ACCESS = ("local", "tether")


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def validate_record(
    rec: dict, *, filename: str | None = None, analysis_dir: Path | None = None
) -> list[str]:
    """Return a list of human-readable problems; empty list means the record is well-formed.

    Returning problems instead of raising lets the CLI report every fault in one
    pass -- a half-validated ledger is worse than an unvalidated one because it
    invites the reader to assume the rest was checked.
    """
    problems: list[str] = []

    for key in REQUIRED_TOP:
        if key not in rec:
            problems.append(f"missing required key: {key}")
    if problems:
        return problems

    if rec["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version {rec['schema_version']} != {SCHEMA_VERSION}")
    if not _ID_RE.match(rec["dataset_id"]):
        problems.append(f"dataset_id {rec['dataset_id']!r} is not <exp>/<suite>/<name>")
    if rec["kind"] not in KNOWN_KINDS:
        problems.append(f"kind {rec['kind']!r} not in {KNOWN_KINDS}")
    if rec["status"] not in KNOWN_STATUS:
        problems.append(f"status {rec['status']!r} not in {KNOWN_STATUS}")

    if filename is not None:
        expected = _filename_for(rec["dataset_id"])
        if filename != expected:
            problems.append(
                f"filename {filename!r} does not match dataset_id (expected {expected!r})"
            )

    auth = rec["authority"]
    for key in ("node", "path", "access"):
        if key not in auth:
            problems.append(f"authority.{key} missing")
    if auth.get("access") not in KNOWN_ACCESS:
        problems.append(
            f"authority.access {auth.get('access')!r} not in {KNOWN_ACCESS}"
        )
    if (
        isinstance(auth.get("path"), str)
        and auth["access"] == "tether"
        and not auth["path"].startswith("/")
    ):
        # A relative path on a remote node is not resolvable by any reader.
        problems.append("authority.path must be absolute when access is 'tether'")

    integ = rec["integrity"]
    sha = integ.get("sha256")
    if not isinstance(sha, str) or not _SHA256_RE.match(sha):
        problems.append(f"integrity.sha256 {sha!r} is not 64 lowercase hex chars")
    for key in ("size_bytes", "file_count"):
        val = integ.get(key)
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            problems.append(f"integrity.{key} must be a positive int, got {val!r}")

    prov = rec["provenance"]
    for key in ("produced_by", "measured_at", "measured_by"):
        if not prov.get(key):
            problems.append(f"provenance.{key} missing or empty")

    for key in ("caveats", "consumers"):
        if key in rec and not isinstance(rec[key], list):
            problems.append(f"{key} must be a list when present")

    # A dangling analysis pointer is worse than none: it promises an attributable
    # figure set and delivers a missing directory.
    task = rec.get("analysis_task")
    if task is not None:
        task_dir = (analysis_dir or ANALYSIS_DIR) / str(task)
        if not (task_dir / MANIFEST_NAME).is_file():
            problems.append(
                f"analysis_task {task!r} has no {task_dir.name}/{MANIFEST_NAME}"
            )

    return problems


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


def _filename_for(dataset_id: str) -> str:
    return dataset_id.replace(ID_SEP, FILE_SEP) + ".json"


def record_path_for(dataset_id: str, *, records_dir: Path | None = None) -> Path:
    """Return the on-disk ledger path a dataset_id maps to (whether or not it exists)."""
    return (records_dir or RECORDS_DIR) / _filename_for(dataset_id)


def load_record(path: str | Path) -> dict:
    """Read one ledger record and validate it. Raises ValueError on any problem."""
    path = Path(path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_record(rec, filename=path.name)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return rec


def load_all(records_dir: Path | None = None) -> list[dict]:
    """Read every ledger record, validated, ordered by dataset_id."""
    directory = records_dir or RECORDS_DIR
    recs = [
        load_record(p)
        for p in sorted(directory.glob("*.json"))
        if not p.name.startswith("_")
    ]
    return sorted(recs, key=lambda r: r["dataset_id"])


def find(
    *,
    experiment: str | None = None,
    suite: str | None = None,
    kind: str | None = None,
    status: str | None = "authoritative",
    records_dir: Path | None = None,
) -> list[dict]:
    """Filter the ledger. ``None`` means "do not constrain on this axis"."""
    out = []
    for rec in load_all(records_dir):
        if experiment is not None and experiment not in rec["experiment"]:
            continue
        if suite is not None and rec.get("suite") != suite:
            continue
        if kind is not None and rec["kind"] != kind:
            continue
        if status is not None and rec["status"] != status:
            continue
        out.append(rec)
    return out


# ------------------------------------------------------------------
# Analysis manifests
# ------------------------------------------------------------------

MANIFEST_REQUIRED = (
    "schema_version",
    "task",
    "title",
    "policy",
    "collected_at",
    "files",
)
MANIFEST_FILE_REQUIRED = ("name", "sha256", "size_bytes", "source")


def validate_analysis_manifest(manifest: dict, *, task: str | None = None) -> list[str]:
    """Return problems with one ``analysis/<task>/MANIFEST.json``; empty means well-formed."""
    problems: list[str] = []
    for key in MANIFEST_REQUIRED:
        if key not in manifest:
            problems.append(f"missing required key: {key}")
    if problems:
        return problems

    if task is not None and manifest["task"] != task:
        problems.append(f"task {manifest['task']!r} does not match directory {task!r}")
    if manifest["policy"] != "copy":
        problems.append(
            f"policy {manifest['policy']!r} must be 'copy' (collection never moves)"
        )
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        problems.append("files must be a non-empty list")
        return problems

    seen: set[str] = set()
    for idx, entry in enumerate(manifest["files"]):
        for key in MANIFEST_FILE_REQUIRED:
            if not entry.get(key):
                problems.append(f"files[{idx}].{key} missing or empty")
        name = entry.get("name")
        if name in seen:
            problems.append(f"files[{idx}].name {name!r} is duplicated")
        seen.add(name)
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not _SHA256_RE.match(sha):
            problems.append(
                f"files[{idx}].sha256 {sha!r} is not 64 lowercase hex chars"
            )
        size = entry.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            problems.append(
                f"files[{idx}].size_bytes must be a positive int, got {size!r}"
            )
    return problems


def analysis_tasks(analysis_dir: Path | None = None) -> list[str]:
    """Task slugs that have a manifest, sorted."""
    directory = analysis_dir or ANALYSIS_DIR
    if not directory.is_dir():
        return []
    return sorted(d.name for d in directory.iterdir() if (d / MANIFEST_NAME).is_file())


def load_analysis_manifest(task: str, analysis_dir: Path | None = None) -> dict:
    """Read one task manifest, validated. Raises ValueError on any problem."""
    path = (analysis_dir or ANALYSIS_DIR) / task / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_analysis_manifest(manifest, task=task)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return manifest


def load_all_analysis(analysis_dir: Path | None = None) -> list[dict]:
    """Every task manifest, validated, ordered by task slug."""
    return [
        load_analysis_manifest(t, analysis_dir) for t in analysis_tasks(analysis_dir)
    ]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _fmt_row(rec: dict) -> str:
    content = rec.get("content", {})
    extra = ""
    if "trajectories" in content:
        extra = (
            f"  traj={content['trajectories']:<5} entries={content.get('entries', '?')}"
        )
    task = rec.get("analysis_task")
    tail = f"  analysis={task}" if task else ""
    return f"{rec['dataset_id']:<48} {rec['authority']['node']:<15}{extra}{tail}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Authoritative dataset ledger")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="list records")
    p_ls.add_argument("--experiment")
    p_ls.add_argument("--suite")
    p_ls.add_argument("--kind")
    p_ls.add_argument(
        "--all-status", action="store_true", help="include superseded records"
    )

    p_show = sub.add_parser("show", help="print one record")
    p_show.add_argument("dataset_id")

    sub.add_parser("validate", help="validate every record and every analysis manifest")

    p_an = sub.add_parser("analysis", help="list collected analysis tasks")
    p_an.add_argument("task", nargs="?", help="show one task's manifest")

    args = ap.parse_args(argv)

    if args.cmd == "ls":
        recs = find(
            experiment=args.experiment,
            suite=args.suite,
            kind=args.kind,
            status=None if args.all_status else "authoritative",
        )
        for rec in recs:
            print(_fmt_row(rec))
        print(f"\n{len(recs)} record(s)")
        return 0

    if args.cmd == "show":
        path = record_path_for(args.dataset_id)
        if not path.exists():
            print(f"no such dataset_id: {args.dataset_id}", file=sys.stderr)
            return 1
        print(json.dumps(load_record(path), indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "analysis":
        if args.task:
            print(
                json.dumps(
                    load_analysis_manifest(args.task), indent=2, ensure_ascii=False
                )
            )
            return 0
        for manifest in load_all_analysis():
            count = len(manifest["files"])
            total = sum(f["size_bytes"] for f in manifest["files"])
            print(
                f"{manifest['task']:<24} {count:>3} file(s)  {total / 1024:>9.1f} KiB  {manifest['title']}"
            )
        return 0

    failures = 0
    for path in sorted(RECORDS_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        problems = validate_record(
            json.loads(path.read_text(encoding="utf-8")), filename=path.name
        )
        if problems:
            failures += 1
            print(f"FAIL {path.name}")
            for problem in problems:
                print(f"       - {problem}")
        else:
            print(f"ok   {path.name}")

    for task in analysis_tasks():
        path = ANALYSIS_DIR / task / MANIFEST_NAME
        problems = validate_analysis_manifest(
            json.loads(path.read_text(encoding="utf-8")), task=task
        )
        if problems:
            failures += 1
            print(f"FAIL analysis/{task}/{MANIFEST_NAME}")
            for problem in problems:
                print(f"       - {problem}")
        else:
            print(f"ok   analysis/{task}/{MANIFEST_NAME}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
