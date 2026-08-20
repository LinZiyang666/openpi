"""Copy analysis output into ``analysis/<task>/`` and write its MANIFEST.

A collected figure whose origin nobody recorded is an orphan, and a registry
full of orphans is worse than no registry -- it looks authoritative while being
unattributable. So collection and manifest-writing are one operation here: you
cannot get a file into a task directory through this tool without its source
path and digest landing in ``MANIFEST.json`` alongside it.

Copy, never move. Experiment ``analysis/*.md`` reports link their figures by
relative path, so moving the file out from under a published report silently
breaks it.

Hand-written ``description`` fields already present in a manifest survive a
re-run: the tool owns ``sha256`` / ``size_bytes`` / ``source``, the human owns
the prose.

Public interface: ``collect``, ``write_manifest``, ``refresh``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

from exp.data_authority.registry import ANALYSIS_DIR, MANIFEST_NAME, REPO_ROOT
from exp.data_authority.verify import sha256_file

MANIFEST_SCHEMA_VERSION = 1


# ------------------------------------------------------------------
# Manifest assembly
# ------------------------------------------------------------------


def _describe(task_dir: Path, rel: str, source: str | None, previous: dict) -> dict:
    """One manifest entry. ``previous`` supplies a description if one was written."""
    full = task_dir / rel
    entry = {
        "name": rel,
        "sha256": sha256_file(full),
        "size_bytes": full.stat().st_size,
        "source": source if source is not None else previous.get("source"),
    }
    if previous.get("description"):
        entry["description"] = previous["description"]
    return entry


def write_manifest(
    task_dir: Path,
    *,
    task: str,
    title: str | None = None,
    source_experiment: str | None = None,
    collected_at: str | None = None,
    sources: dict[str, str] | None = None,
) -> dict:
    """Re-hash every file under ``task_dir`` and write ``MANIFEST.json``.

    ``sources`` maps a relative name to the path it was copied from; names not
    present there keep whatever source the previous manifest recorded, which is
    what makes a plain refresh non-destructive.
    """
    manifest_path = task_dir / MANIFEST_NAME
    old = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    old_files = {f["name"]: f for f in old.get("files", [])}

    rels = sorted(
        str(p.relative_to(task_dir))
        for p in task_dir.rglob("*")
        if p.is_file() and not p.is_symlink() and p.name != MANIFEST_NAME
    )
    files = [
        _describe(task_dir, rel, (sources or {}).get(rel), old_files.get(rel, {}))
        for rel in rels
    ]

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task": task,
        "title": title or old.get("title") or task,
        "source_experiment": source_experiment or old.get("source_experiment"),
        "policy": "copy",
        "collected_at": collected_at
        or old.get("collected_at")
        or datetime.date.today().isoformat(),
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


# ------------------------------------------------------------------
# Collection
# ------------------------------------------------------------------


def collect(
    task: str,
    specs: list[str],
    *,
    title: str | None = None,
    source_experiment: str | None = None,
    force: bool = False,
    analysis_dir: Path | None = None,
) -> dict:
    """Copy each ``<src>`` or ``<src>:<dest-rel>`` spec into the task directory.

    Raises FileExistsError rather than overwriting, unless ``force`` -- an
    accidental overwrite here would replace an authoritative copy with an
    unrelated file of the same basename, which is exactly the failure the
    digests are meant to make impossible.
    """
    task_dir = (analysis_dir or ANALYSIS_DIR) / task
    task_dir.mkdir(parents=True, exist_ok=True)

    sources: dict[str, str] = {}
    for spec in specs:
        src_text, _, dest_rel = spec.partition(":")
        src = Path(src_text)
        if not src.is_file():
            raise FileNotFoundError(f"not a regular file: {src}")
        rel = dest_rel or src.name
        dest = task_dir / rel
        if dest.exists() and not force:
            raise FileExistsError(f"{dest} exists; pass --force to replace")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        try:
            sources[rel] = str(src.resolve().relative_to(REPO_ROOT))
        except ValueError:
            # Outside the repo (e.g. pulled from a remote node): keep it verbatim
            # so the manifest still says where the bytes came from.
            sources[rel] = str(src)

    return write_manifest(
        task_dir,
        task=task,
        title=title,
        source_experiment=source_experiment,
        sources=sources,
    )


def refresh(task: str, *, analysis_dir: Path | None = None) -> dict:
    """Re-hash an existing task directory without copying anything in."""
    task_dir = (analysis_dir or ANALYSIS_DIR) / task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"no such task directory: {task_dir}")
    return write_manifest(task_dir, task=task)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Collect analysis output into the registry"
    )
    ap.add_argument("task", help="task slug -> analysis/<task>/")
    ap.add_argument("sources", nargs="*", help="<src> or <src>:<dest-rel>")
    ap.add_argument("--title")
    ap.add_argument("--source-experiment")
    ap.add_argument(
        "--force", action="store_true", help="replace existing destination files"
    )
    ap.add_argument(
        "--refresh", action="store_true", help="re-hash the task dir, copy nothing"
    )
    args = ap.parse_args(argv)

    try:
        if args.refresh:
            manifest = refresh(args.task)
        else:
            if not args.sources:
                print(
                    "no sources given (use --refresh to only re-hash)", file=sys.stderr
                )
                return 64
            manifest = collect(
                args.task,
                args.sources,
                title=args.title,
                source_experiment=args.source_experiment,
                force=args.force,
            )
    except (FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"{args.task}: {len(manifest['files'])} file(s) in manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
