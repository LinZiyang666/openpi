"""Serve the point editor for a RoboCasa365 RIT figure spec.

Three routes and nothing else: the page, the spec it edits, and the write that
overwrites that spec in place. Rendering, auditing and export are deliberately
absent -- the page is the figure, and the spec is the only artifact.

The write is atomic and its destination comes from the served directory, never
from the request, so a half-posted body cannot truncate a good spec.

Takes no arguments: it serves every spec of this schema in ``analysis/figures``
on a fixed loopback address, and the page switches between them.

Usage:
  uv run python -m exp.robocasa365.edit_rit_figure
"""

from __future__ import annotations

import http.server
import json
import pathlib
import urllib.parse

FIGURES_DIR = pathlib.Path(__file__).with_name("analysis") / "figures"
HOST = "127.0.0.1"
PORT = 8765
_HTML = pathlib.Path(__file__).with_name("rit_figure_editor.html")
_SCHEMA = "robocasa365.rit_figure/v1"
_MAX_BODY = 16 << 20


def discover(figures_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Figure id -> spec path, for every spec of this schema in the directory."""
    out: dict[str, pathlib.Path] = {}
    for path in sorted(figures_dir.glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(spec, dict) and spec.get("schema") == _SCHEMA:
            out[str(spec.get("figure_id") or path.stem)] = path
    return out


def validate(spec: dict, figure_id: str) -> None:
    """Enough of a check that a broken page state cannot land on disk."""
    if spec.get("schema") != _SCHEMA:
        raise ValueError(f"schema must be {_SCHEMA!r}")
    if spec.get("figure_id") != figure_id:
        raise ValueError("figure_id in the body does not match the target")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list")
    n_ep = int(spec.get("episodes_per_task") or 0)
    if n_ep <= 0:
        raise ValueError("episodes_per_task must be positive")
    for series in spec.get("series", []):
        for point in series.get("points", []):
            for task, c in point.get("per_task", {}).items():
                if task not in tasks:
                    raise ValueError(f"point {point.get('id')!r} names unknown task {task!r}")
                y = float(c["y"])
                if abs(y * n_ep - round(y * n_ep)) > 1e-6:
                    raise ValueError(
                        f"point {point.get('id')!r} task {task!r}: y*{n_ep} = {y * n_ep} "
                        "is not an integer"
                    )


def write_spec(spec: dict, path: pathlib.Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def make_handler(figures_dir: pathlib.Path) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "rit-figure-editor/1"

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _target(self, query: dict) -> tuple[str, pathlib.Path] | None:
            figures = discover(figures_dir)
            if not figures:
                return None
            wanted = (query.get("id") or [None])[0]
            if wanted is None:
                wanted = next(iter(figures))
            path = figures.get(wanted)
            return (wanted, path) if path else None

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                self._send(200, _HTML.read_bytes(), "text/html; charset=utf-8")
                return
            if url.path == "/figures":
                self._json(200, {"figures": sorted(discover(figures_dir))})
                return
            if url.path == "/spec":
                target = self._target(query)
                if target is None:
                    self._json(404, {"error": "no figure spec found"})
                    return
                figure_id, path = target
                self._json(200, {"figure_id": figure_id,
                                 "spec": json.loads(path.read_text(encoding="utf-8"))})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            url = urllib.parse.urlparse(self.path)
            if url.path != "/save":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > _MAX_BODY:
                self._json(413, {"error": "bad body length"})
                return
            try:
                spec = json.loads(self.rfile.read(length).decode("utf-8"))
            except ValueError as exc:
                self._json(400, {"error": f"bad JSON: {exc}"})
                return
            target = self._target(urllib.parse.parse_qs(url.query))
            if target is None:
                self._json(404, {"error": "no figure spec found"})
                return
            figure_id, path = target
            try:
                validate(spec, figure_id)
            except (ValueError, KeyError, TypeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            write_spec(spec, path)
            self._json(200, {"saved": str(path)})

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler


def main() -> None:
    found = discover(FIGURES_DIR)
    if not found:
        raise SystemExit(f"no {_SCHEMA} spec under {FIGURES_DIR}")
    server = http.server.ThreadingHTTPServer((HOST, PORT), make_handler(FIGURES_DIR))
    print(f"editing {', '.join(sorted(found))}")
    print(f"open http://{HOST}:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
