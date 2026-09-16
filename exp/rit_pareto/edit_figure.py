"""Serve the interactive editor for the RIT-Pareto figures -- stage three of the chain.

Takes no arguments: it serves every figure spec in ``analysis/figures`` (the four
published frontiers) and the page switches between them. Points can be dragged,
annotations nudged and coordinates typed in; the Pareto frontier of every series
is recomputed live -- it is derived, never edited.

Two writes are offered, both onto the files the selected figure already owns:

``Save``    overwrites ``<figure_id>.json``
``Export``  overwrites ``<figure_id>.json`` and re-renders ``.png`` / ``.pdf``

A series that points at another figure is drawn from that figure's spec and can be
edited right here: on Save the edited points are written back into the figure that
owns them, so one set of points sits behind every line that shows them. Exporting
re-renders every spec written plus whichever figures reference one of them, so no
published png can quietly disagree with the spec behind it.

The export runs ``render_figure`` server-side rather than exporting the browser's
SVG, so what lands on disk comes off the same matplotlib path as an unedited
render; the page itself is a preview, not a second renderer. Destinations come
from the served directory and never from the request, and each write is validated
and atomic, so a broken page state cannot truncate a good spec.
"""

from __future__ import annotations

import argparse
import contextlib
import http.server
import json
import math
import os
import pathlib
import platform
import subprocess
import threading
import urllib.parse
import webbrowser

from exp.rit_pareto import render_figure

#: The four published figures live here; the editor serves whatever it finds.
FIGURES_DIR = pathlib.Path(__file__).with_name("analysis") / "figures"

_HTML = pathlib.Path(__file__).with_name("figure_editor.html")
_MAX_BODY = 32 << 20


# ------------------------------------------------------------------
# Figure discovery
# ------------------------------------------------------------------

def discover(figures_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Figure id -> spec path for every readable spec in the directory."""
    out = {}
    for path in sorted(figures_dir.glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if spec.get("schema") == render_figure.SCHEMA and spec.get("figure_id") == path.stem:
            out[path.stem] = path
    return out


def dependents(figure_id: str, figures: dict[str, pathlib.Path]) -> list[pathlib.Path]:
    """Specs whose series read their points from ``figure_id``."""
    out = []
    for other_id, path in figures.items():
        if other_id == figure_id:
            continue
        spec = json.loads(path.read_text(encoding="utf-8"))
        if any(s.get("source_figure") == figure_id for s in spec.get("series", [])):
            out.append(path)
    return out


def resolved_refs(spec: dict, base_dir: pathlib.Path) -> dict[str, list[dict]]:
    """Points the page must draw for each referenced series, keyed by series key."""
    return {s["key"]: render_figure.series_points(s, base_dir)
            for s in spec.get("series", []) if "source_figure" in s}


# ------------------------------------------------------------------
# Validation and writes
# ------------------------------------------------------------------

def _check_points(points, where: str) -> None:
    if not isinstance(points, list):
        raise ValueError(f"{where}: points must be a list")
    for point in points:
        for axis in ("x", "y"):
            value = point.get(axis) if isinstance(point, dict) else None
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"point {point.get('id') if isinstance(point, dict) else point!r}: "
                                 f"{axis} must be a finite number")


def validate(spec: dict, expect_id: str | None = None) -> None:
    """Reject anything that would render badly or overwrite the wrong figure."""
    if not isinstance(spec, dict) or spec.get("schema") != render_figure.SCHEMA:
        raise ValueError(f"schema must be {render_figure.SCHEMA!r}")
    if expect_id is not None and spec.get("figure_id") != expect_id:
        raise ValueError(f"figure_id {spec.get('figure_id')!r} does not match {expect_id!r}")
    if not isinstance(spec.get("series"), list) or not spec["series"]:
        raise ValueError("series must be a non-empty list")
    for series in spec["series"]:
        if "source_figure" in series and series.get("points"):
            raise ValueError(f"series {series.get('key')!r} references {series['source_figure']!r} "
                             "and must not carry its own points")
        _check_points(series.get("points", []), f"series {series.get('key')!r}")


def _atomic_write(spec: dict, path: pathlib.Path) -> pathlib.Path:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(spec, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def save_spec(spec: dict, spec_path: pathlib.Path) -> pathlib.Path:
    """Validate and atomically overwrite the spec file."""
    validate(spec, spec_path.stem)
    return _atomic_write(spec, spec_path)


def write_back_refs(spec: dict, refs: dict, figures: dict[str, pathlib.Path]) -> list[pathlib.Path]:
    """Store the edited points of referenced series in the figures that own them.

    Only series the host references are touched, and only their ``points`` are
    replaced; the owner must hold the points itself (a reference chain is refused).
    Returns the owner specs that actually changed.
    """
    written = []
    for series in spec.get("series", []):
        owner_id, key = series.get("source_figure"), series.get("key")
        if owner_id is None or key not in refs:
            continue
        if owner_id not in figures:
            raise ValueError(f"series {key!r} references unknown figure {owner_id!r}")
        owner = json.loads(figures[owner_id].read_text(encoding="utf-8"))
        target = next((t for t in owner.get("series", []) if t.get("key") == series.get("source_series")), None)
        if target is None or "source_figure" in target:
            raise ValueError(f"{owner_id} has no editable series {series.get('source_series')!r}")
        _check_points(refs[key], f"series {key!r}")
        if target.get("points") == refs[key]:
            continue
        target["points"] = refs[key]
        written.append(_atomic_write(owner, figures[owner_id]))
    return written


def save_figure(spec: dict, refs: dict, spec_path: pathlib.Path,
                figures: dict[str, pathlib.Path]) -> list[pathlib.Path]:
    """Write the host spec and every owner spec its edited references belong to."""
    validate(spec, spec_path.stem)
    owners = write_back_refs(spec, refs, figures)
    return [save_spec(spec, spec_path), *owners]


def is_published(spec_path: pathlib.Path) -> bool:
    """Whether this spec already has figures on disk.

    A spec that only carries points for other figures to reference (and was never
    rendered itself) is not a figure: it is not offered for editing, and it must not
    gain a png just because something referenced it. Re-rendering only ever refreshes
    figures that already exist.
    """
    return (spec_path.with_suffix(".png").is_file() or spec_path.with_suffix(".pdf").is_file())


def published(figures_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """The specs that are actually figures -- what the page offers in its picker."""
    return {fid: path for fid, path in discover(figures_dir).items() if is_published(path)}


def export_figure(spec: dict, refs: dict, spec_path: pathlib.Path,
                  figures: dict[str, pathlib.Path]) -> list[pathlib.Path]:
    """Save, then refresh the figures of every spec written and of whatever references one."""
    saved = save_figure(spec, refs, spec_path, figures)
    to_render = list(saved)
    for path in saved:
        for other in dependents(path.stem, figures):
            if other not in to_render:
                to_render.append(other)
    written = list(saved)
    for path in to_render:
        if path == spec_path or is_published(path):
            written.extend(render_figure.render_file(path))
    return written


# ------------------------------------------------------------------
# HTTP surface
# ------------------------------------------------------------------

def make_handler(figures_dir: pathlib.Path) -> type[http.server.BaseHTTPRequestHandler]:
    """Handler bound to one directory; the client picks a figure by id, never by path."""

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "rit_pareto_figure_editor"

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def _selected(self, query: dict) -> tuple[str, pathlib.Path] | None:
            """The requested figure, or None (a 404 is already sent) if it is not served."""
            figures = published(figures_dir)
            wanted = (query.get("id") or [None])[0]
            if wanted is None and figures:
                wanted = next(iter(figures))
            if wanted not in figures:
                self._json(404, {"error": f"unknown figure {wanted!r}"})
                return None
            return wanted, figures[wanted]

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            route = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(route.query)
            if route.path in ("/", "/index.html"):
                self._send(200, _HTML.read_bytes(), "text/html; charset=utf-8")
            elif route.path == "/api/figures":
                figures = published(figures_dir)
                self._json(200, {"figures": [
                    {"id": fid, "title": json.loads(p.read_text(encoding="utf-8")).get("title", fid)}
                    for fid, p in figures.items()]})
            elif route.path == "/api/figure":
                picked = self._selected(query)
                if picked is None:
                    return
                spec = json.loads(picked[1].read_text(encoding="utf-8"))
                self._json(200, {"spec": spec, "refs": resolved_refs(spec, figures_dir)})
            elif route.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            route = urllib.parse.urlparse(self.path)
            if route.path == "/api/save":
                # Owner 2026-09-15: plain JSON saves are switched off; Export is the only write.
                self._json(403, {"error": "Save JSON is disabled; use Export"})
                return
            if route.path not in ("/api/save", "/api/export"):
                self._json(404, {"error": "not found"})
                return
            picked = self._selected(urllib.parse.parse_qs(route.query))
            if picked is None:
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._json(411, {"error": "Content-Length required"})
                return
            if length > _MAX_BODY:
                self._json(413, {"error": "body too large"})
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                spec, refs = body.get("spec", body), body.get("refs") or {}
                figures = discover(figures_dir)
                written = (export_figure if route.path == "/api/export" else save_figure)(
                    spec, refs, picked[1], figures)
            except (ValueError, UnicodeDecodeError, AttributeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {"written": [str(p) for p in written]})

        def log_message(self, fmt: str, *args) -> None:
            # One quiet line per request; the default logs are noisy for a single-user tool.
            print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)

    return Handler


def open_browser(url: str) -> bool:
    """Best-effort launch. Under WSL no Linux browser exists, but explorer.exe reaches
    the Windows one; elsewhere fall back to ``webbrowser`` only if one is registered,
    so a headless host prints the URL instead of a page of xdg-open errors."""
    if "microsoft" in platform.uname().release.lower():
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(["explorer.exe", url], check=False, timeout=15,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    try:
        webbrowser.get()
    except webbrowser.Error:
        return False
    return webbrowser.open(url)


def serve(figures_dir: pathlib.Path = FIGURES_DIR, host: str = "127.0.0.1", port: int = 0,
          launch_browser: bool = True) -> None:
    """Run the editor until interrupted; port 0 picks a free one and prints the URL."""
    figures = discover(figures_dir)
    if not figures:
        raise SystemExit(f"no figure specs in {figures_dir} -- run build_figure first")
    server = http.server.ThreadingHTTPServer((host, port), make_handler(figures_dir))
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"figures: {', '.join(figures)}")
    print(f"editor : {url}   (Ctrl-C to stop)")
    if launch_browser:
        threading.Thread(target=lambda: open_browser(url), daemon=True).start()
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    server.server_close()
    print("editor stopped")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port")
    ap.add_argument("--no-browser", action="store_true", help="print the URL instead of opening it")
    args = ap.parse_args()
    serve(port=args.port, launch_browser=not args.no_browser)


if __name__ == "__main__":
    main()
