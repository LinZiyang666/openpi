"""Bounded tether transfers and local verified payload installation."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .build_client_bundle import sha256, verified_members
from .sync_back import atomic_bytes


def tether_transfer(argv, *, lane, runner=subprocess.run, sleeper=time.sleep, attempts=8,
                    retry_seconds=0, clock=None, on_event=None):
    if lane not in ("local", "tether") or not 1 <= attempts <= 8:
        raise ValueError("explicit authorized transfer lane and bounded attempts required")
    if retry_seconds:
        if lane != 'tether':
            raise ValueError('transport retry requires the tether lane')
        from .tether_retry import retry
        count = 0
        def operation():
            nonlocal count
            count += 1
            from .tether_retry import current_journal
            if current_journal.get():
                endpoint=next((str(a).split(':',1)[0] for a in argv[2:4] if ':' in str(a)),None)
                current_journal.get().record_call(' '.join(map(str,argv[:2])),endpoint)
            result = runner(list(argv), capture_output=True, text=True)
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, argv, result.stdout, result.stderr)
            return dict(status='ok', attempts=count, lane=lane)
        return retry(operation,label=' '.join(map(str,argv[:4])),seconds=retry_seconds,
                     sleeper=sleeper,clock=clock,on_event=on_event,replay_safe=True)
    for attempt in range(attempts):
        result = runner(list(argv), capture_output=True, text=True)
        if result.returncode == 0:
            return dict(status="ok", attempts=attempt + 1, lane=lane)
        if result.returncode != 75 or "too_many_in_flight" not in result.stderr or attempt == attempts-1:
            raise subprocess.CalledProcessError(result.returncode, argv, result.stdout, result.stderr)
        sleeper(min(2 ** (attempt+1), 30))


def install_payload(archive, destination, expected_sha256):
    if sha256(archive) != expected_sha256:
        raise ValueError("client archive SHA mismatch")
    manifest, content = verified_members(archive, "PAYLOAD.json")
    destination = Path(destination)
    marker = destination / "PAYLOAD.json"
    if marker.exists():
        import json
        if json.loads(marker.read_text()) != manifest:
            raise ValueError("different payload already installed; use a new SHA directory")
    destination.mkdir(parents=True, exist_ok=True)
    # Validate every target before installation; reject an unexpected existing file.
    for name in content:
        target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()) or target.is_symlink():
            raise ValueError("escaped payload install path")
        if target.exists() and sha256(target) != manifest["files"][name]:
            raise ValueError("payload target changed")
    for name, data in content.items():
        if not (destination / name).exists():
            atomic_bytes(destination / name, data)
    from .build_client_bundle import json_bytes
    atomic_bytes(marker, json_bytes(manifest))
    return manifest
