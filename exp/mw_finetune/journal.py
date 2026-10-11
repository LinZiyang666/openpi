"""Durable JSONL loading; recover only an interrupted final append, preserving it."""
import json
import os
from pathlib import Path
import shutil
import time


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".write." + str(time.time_ns()))
    with temporary.open("x") as out:
        json.dump(value, out, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def read_journal(path, recover_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    valid_bytes = 0
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not recover_tail or i != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"invalid journal JSON at line {i+1}: {path}") from None
            suffix = str(time.time_ns())
            backup = path.with_name(path.name + ".torn." + suffix)
            shutil.copy2(path, backup)
            repaired = path.with_name(path.name + ".repair." + suffix)
            with repaired.open("xb") as out:
                out.write(raw[:valid_bytes])
                out.flush()
                os.fsync(out.fileno())
            os.replace(repaired, path)
            print(f"Recovered interrupted final journal append; original preserved at {backup}", flush=True)
            return rows
        valid_bytes += len(line)
    if raw and not raw.endswith(b"\n") and recover_tail:
        with path.open("ab") as out:
            out.write(b"\n")
            out.flush()
            os.fsync(out.fileno())
    return rows
