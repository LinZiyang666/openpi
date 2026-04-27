"""Per-trajectory F1b factor explorer (interactive web UI).

Loads `cp1_mean_pool.pkl` from the libero_spatial cache artifacts (any of
the six pkls would do — factors are byte-identical across them, see
`logs/libero_spatial_factor_artifact_rebuild.log.md` §6 sanity), bakes
the per-entry factor data into a single static HTML and serves it on
`http://127.0.0.1:8000/factor_explorer.html` so factor sequences can be
inspected without round-tripping through matplotlib.

Layout:
- Left sidebar: 21 windows in one vertical column, grouped into three
  families (pure-future / pure-past / symmetric) separated by a small
  gap; each window is a checkbox.
- Top: trajectory dropdown — single-select, shows
  ``[idx] traj_id (T=N)`` per option, sorted by id.
- Main: 8 Plotly subplots in a 4-row x 2-col grid. Left column = F1b-A
  (action-side), right column = F1b-T (state-side). Rows are
  ``dir / cum_disp / jerk / curv_radius`` top-to-bottom.
- Color: HSL hue rotation across the active window selection — 1 color
  per window, distinct regardless of how many were picked.
- Legend: rendered ONCE in a strip above the plots; per-plot legend off.
- NaN handling: serialized as JSON ``null``; Plotly draws gaps where
  large windows can't fit at episode boundaries.

Run:
    uv run python exp/common/analysis/factor_explorer.py
    # then open http://127.0.0.1:8000/factor_explorer.html
"""

from __future__ import annotations

import http.server
import json
import math
import os
import pickle
import socketserver
import sys
from collections import defaultdict
from pathlib import Path

PKL = "exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool.pkl"
OUT = Path(__file__).parent / "factor_explorer.html"
PORT = 8000

DESCRIPTORS = ["dir", "cum_disp", "jerk", "curv_radius"]   # display order

WINDOWS_PURE_FUTURE = [(0, k) for k in range(1, 8)]
WINDOWS_PURE_PAST = [(k, 0) for k in range(1, 8)]
WINDOWS_SYM = [(k, k) for k in range(1, 8)]
ALL_WINDOWS = WINDOWS_PURE_FUTURE + WINDOWS_PURE_PAST + WINDOWS_SYM


def _win_label(p: int, f: int) -> str:
    return f"p{p}_f{f}"


def load_data() -> dict:
    """Load pkl, group by trajectory_id, sort by step_idx, extract per-key
    factor sequences as JSON-friendly null-or-float lists.
    """
    with open(PKL, "rb") as fh:
        art = pickle.load(fh)

    by_traj = defaultdict(list)
    for entry in art["entries"]:
        by_traj[entry.trajectory_id].append(entry)

    keys: list[str] = []
    for desc in DESCRIPTORS:
        for src in ("a", "t"):
            for (p, f) in ALL_WINDOWS:
                keys.append(f"f1b_{src}_{desc}__{_win_label(p, f)}")

    out: dict = {}
    for tid, entries in by_traj.items():
        entries.sort(key=lambda e: e.step_idx)
        T = len(entries)
        factors_by_key: dict[str, list] = {}
        for k in keys:
            seq = []
            for e in entries:
                pf = e.payload.factors
                if pf is None:
                    seq.append(None)
                    continue
                v = pf.get(k)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    seq.append(None)
                else:
                    seq.append(float(v))
            factors_by_key[k] = seq
        out[tid] = {"T": T, "steps": list(range(T)), "factors": factors_by_key}
    return out


def build_html(data: dict, default_traj: str) -> None:
    payload = json.dumps(
        {
            "trajectories": data,
            "default_traj": default_traj,
            "default_windows": ["p3_f3", "p0_f5", "p5_f5"],
            "descriptors": DESCRIPTORS,
            "windows_pure_future": [
                {"label": _win_label(p, f), "p": p, "f": f}
                for (p, f) in WINDOWS_PURE_FUTURE
            ],
            "windows_pure_past": [
                {"label": _win_label(p, f), "p": p, "f": f}
                for (p, f) in WINDOWS_PURE_PAST
            ],
            "windows_sym": [
                {"label": _win_label(p, f), "p": p, "f": f}
                for (p, f) in WINDOWS_SYM
            ],
        },
        separators=(",", ":"),
    )

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", payload)
    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT}  ({size_kb:.1f} KB)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Libero Spatial F1b Factor Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; padding: 0; overflow: hidden; }
body { font-family: -apple-system, sans-serif; color: #222; font-size: 12px; }
/* Top-level grid: sidebar | main, full viewport height */
#layout { display: grid; grid-template-columns: 180px 1fr; gap: 12px; padding: 8px; height: 100vh; }
#sidebar { font-size: 11px; overflow-y: auto; }
#sidebar h3 { margin: 0 0 6px 0; font-size: 13px; }
.window-group { margin-bottom: 10px; }
.window-group h4 { margin: 2px 0; font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }
.window-checkbox { display: block; margin: 1px 0; cursor: pointer; }
.window-checkbox input { margin-right: 3px; }
/* Main area is itself a vertical grid: header | legend | colheader | plots(flex) | footer */
#main { display: grid; grid-template-rows: auto auto auto 1fr auto; gap: 4px; min-height: 0; }
#header { padding: 2px 0; border-bottom: 1px solid #ddd; }
#header label { font-weight: 600; }
#header select { font-family: monospace; padding: 2px; min-width: 420px; }
#traj-info { margin-left: 12px; color: #666; }
#legend-strip { padding: 4px 8px; background: #f7f7f7; border: 1px solid #e0e0e0; border-radius: 3px; min-height: 22px; }
.legend-item { display: inline-block; margin-right: 10px; }
.legend-swatch { display: inline-block; width: 16px; height: 3px; vertical-align: middle; margin-right: 3px; border-radius: 1px; }
.column-headers { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.column-headers div { font-weight: 600; padding: 2px 6px; background: #eee; border-radius: 2px; text-align: center; font-size: 11px; }
/* Plots fill remaining vertical space; each plot height = available / 4 */
#plots { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: repeat(4, 1fr); gap: 6px; min-height: 0; }
.plot { border: 1px solid #eee; border-radius: 3px; cursor: crosshair; min-height: 0; }
/* Footer = descref + video, fixed-ish height */
#footer { display: grid; grid-template-columns: 1fr 280px; gap: 8px; max-height: 200px; }
#descref { padding: 6px 8px; background: #fafafa; border: 1px solid #e0e0e0; border-radius: 3px; line-height: 1.3; overflow-y: auto; }
#descref h4 { margin: 0 0 4px 0; font-size: 11px; }
#descref table { border-collapse: collapse; width: 100%; font-size: 10px; }
#descref th, #descref td { border: 1px solid #ddd; padding: 2px 4px; text-align: left; vertical-align: top; }
#descref th { background: #eee; font-weight: 600; }
#descref code { background: #eef; padding: 0 2px; border-radius: 2px; font-size: 10px; }
.ori-safe { background: #d4edda; color: #155724; padding: 0 4px; border-radius: 2px; font-weight: 600; font-size: 10px; }
.ori-risky { background: #f8d7da; color: #721c24; padding: 0 4px; border-radius: 2px; font-weight: 600; font-size: 10px; }
.ori-non { background: #fff3cd; color: #856404; padding: 0 4px; border-radius: 2px; font-weight: 600; font-size: 10px; }
.note { color: #555; margin: 4px 0 0 0; font-size: 10px; }
#video-panel { padding: 6px 8px; background: #fafafa; border: 1px solid #e0e0e0; border-radius: 3px; display: flex; flex-direction: column; }
#video-panel .video-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
#video-panel h4 { margin: 0; font-size: 11px; }
#pip-btn { background: #fff; border: 1px solid #c0c0c0; padding: 2px 8px; border-radius: 3px; font-size: 11px; cursor: pointer; }
#pip-btn:hover { background: #eef; border-color: #88f; }
#pip-btn:active { background: #ddf; }
#pip-btn.active { background: #d4edda; border-color: #28a745; color: #155724; }
#pip-btn:disabled { color: #aaa; cursor: not-allowed; background: #f0f0f0; }
#video-panel video { width: 100%; height: auto; max-height: 140px; background: #000; border-radius: 2px; object-fit: contain; }
#video-panel .video-info { margin-top: 4px; color: #666; font-size: 10px; line-height: 1.3; }
#video-panel .video-missing { padding: 8px; color: #888; font-style: italic; text-align: center; font-size: 11px; }
</style>
</head>
<body>
<div id="layout">
  <div id="sidebar">
    <h3>Windows</h3>
    <div class="window-group">
      <h4>pure-future (0, k)</h4>
      <div id="windows-pure-future"></div>
    </div>
    <div class="window-group">
      <h4>pure-past (k, 0)</h4>
      <div id="windows-pure-past"></div>
    </div>
    <div class="window-group">
      <h4>symmetric (k, k)</h4>
      <div id="windows-sym"></div>
    </div>
  </div>

  <div id="main">
    <div id="header">
      <label>Trajectory:
        <select id="traj-select"></select>
      </label>
      <span id="traj-info"></span>
    </div>

    <div id="legend-strip"></div>

    <div class="column-headers">
      <div>F1b-A &middot; action-side</div>
      <div>F1b-T &middot; state-side</div>
    </div>

    <div id="plots">
      <div id="plot-a-dir"          class="plot"></div> <div id="plot-t-dir"          class="plot"></div>
      <div id="plot-a-cum_disp"     class="plot"></div> <div id="plot-t-cum_disp"     class="plot"></div>
      <div id="plot-a-jerk"         class="plot"></div> <div id="plot-t-jerk"         class="plot"></div>
      <div id="plot-a-curv_radius"  class="plot"></div> <div id="plot-t-curv_radius"  class="plot"></div>
    </div>

    <div id="footer">
     <div id="descref">
      <h4>Descriptor reference</h4>
      <table>
        <thead>
          <tr><th>name</th><th>formula (z-scored, active-DOF subspace)</th><th>physical meaning</th><th>orientation</th><th>higher value &rArr;</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><b>dir</b></td>
            <td>mean cos(v[t], v[t+1])</td>
            <td>direction coherence between consecutive velocity vectors</td>
            <td><span class="ori-safe">safe</span></td>
            <td>more directionally consistent &rarr; smoother trajectory &rarr; SAFE for cache hit; near 1 &rArr; almost a straight line; near 0 / negative &rArr; turning / oscillation</td>
          </tr>
          <tr>
            <td><b>cum_disp</b></td>
            <td>&sum; &Vert;p[t+1] - p[t]&Vert;</td>
            <td>cumulative path length in the window</td>
            <td><span class="ori-non">non-monotonic</span></td>
            <td>more total motion: large &rArr; fast / large stroke; small &rArr; near-stationary. Combined with jerk: small + low-jerk = idle (safe), large + high-jerk = aggressive (risky), large + low-jerk = smooth large motion (safe)</td>
          </tr>
          <tr>
            <td><b>jerk</b></td>
            <td>median &vert;&Delta;&sup2;a / &sigma;&vert;</td>
            <td>magnitude of second-order discrete differences (acceleration change)</td>
            <td><span class="ori-risky">risky</span></td>
            <td>jerkier trajectory &rarr; less smooth &rarr; RISKY for cache hit (cache may be stitching across discontinuities). Median (not mean) absorbs single-frame spikes such as gripper bistable transitions</td>
          </tr>
          <tr>
            <td><b>curv_radius</b></td>
            <td>mean &Vert;p[t] - centroid(window)&Vert;</td>
            <td>geometric dispersion of window points around their centroid in active subspace</td>
            <td><span class="ori-non">non-monotonic</span></td>
            <td>larger spread of window points: very small &rArr; stationary cluster; medium &rArr; arc / circulation; very large &rArr; long straight stroke (points spread out symmetrically). Best read together with cum_disp to disambiguate stationary vs. arc vs. straight</td>
          </tr>
        </tbody>
      </table>
      <p class="note">
        All descriptors are computed in the per-DOF z-scored, active-DOF-restricted subspace (Pi0.5 has 32-dim padded action / state; only ~7-8 DOFs are active). F1b-A (left) reads from <code>payload.action_chunk[0]</code>; F1b-T (right) from <code>query_keys["robot_state"]</code>. Window key suffix <code>__pP_fF</code> means the window covers steps [k-P, k+F] around each entry index k.
      </p>
     </div>
     <div id="video-panel">
      <div class="video-header">
        <h4>Trajectory replay</h4>
        <button id="pip-btn" title="Picture-in-picture (popout floating window)">⧉ PiP</button>
      </div>
      <video id="trajvideo" controls preload="metadata"></video>
      <div class="video-info">
        Click any plot to seek video. Video time-bar drives the cursor lines on all 8 plots (synced by <b>percentage</b>: <code>step / (T-1) = videoTime / videoDuration</code>). <b>PiP</b> pops the video into a floating always-on-top window so you can keep it visible while moving the explorer to another monitor / behind another app.
      </div>
     </div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

// ---- trajectory dropdown ----
const trajSelect = document.getElementById('traj-select');
const trajIds = Object.keys(DATA.trajectories).sort();
trajIds.forEach((tid, i) => {
  const opt = document.createElement('option');
  opt.value = tid;
  opt.textContent = `[${i}] ${tid} (T=${DATA.trajectories[tid].T})`;
  trajSelect.appendChild(opt);
});
trajSelect.value = DATA.default_traj;
trajSelect.addEventListener('change', render);

// ---- window checkboxes ----
function makeWindowCheckboxes(containerId, windows) {
  const c = document.getElementById(containerId);
  windows.forEach(w => {
    const lbl = document.createElement('label');
    lbl.className = 'window-checkbox';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = w.label;
    cb.checked = DATA.default_windows.includes(w.label);
    cb.addEventListener('change', render);
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(' ' + w.label));
    c.appendChild(lbl);
  });
}
makeWindowCheckboxes('windows-pure-future', DATA.windows_pure_future);
makeWindowCheckboxes('windows-pure-past', DATA.windows_pure_past);
makeWindowCheckboxes('windows-sym', DATA.windows_sym);

// ---- color generation: HSL hue rotation, dynamic per N ----
function colorFor(i, n) {
  if (n === 1) return 'hsl(220, 70%, 45%)';
  // Spread 0..330 (avoid wrapping back to red)
  const hue = Math.round((i / (n - 1)) * 330);
  return `hsl(${hue}, 65%, 45%)`;
}

function getActiveWindows() {
  return Array.from(
    document.querySelectorAll('.window-checkbox input:checked'),
  ).map(cb => cb.value);
}

// ---- sync state ----
// Plot ↔ video are synced by PERCENTAGE so any T-step trajectory maps
// to any-duration video uniformly:
//    step / (T - 1) == videoTime / videoDuration
// All 8 plots share a vertical cursor line (Plotly shape) at the current
// step. The video element is the source of truth during playback; plot
// clicks seek the video, which then drives the cursor back via the
// `timeupdate` event.
let _currentTraj = null;
const PLOT_IDS = [];
for (const desc of ['dir', 'cum_disp', 'jerk', 'curv_radius']) {
  for (const src of ['a', 't']) {
    PLOT_IDS.push(`plot-${src}-${desc}`);
  }
}

function cursorShape(stepFloat) {
  // Plotly shape spanning full y-range at x=stepFloat. paper-y so it
  // tracks across whatever the autoscaled y-axis happens to be.
  return [{
    type: 'line',
    xref: 'x', yref: 'paper',
    x0: stepFloat, x1: stepFloat, y0: 0, y1: 1,
    line: { color: '#e63946', width: 1.5, dash: 'solid' },
  }];
}

function setCursor(stepFloat) {
  const shapes = cursorShape(stepFloat);
  PLOT_IDS.forEach(id => {
    Plotly.relayout(id, { shapes }).catch(() => {});
  });
}

function render() {
  const tid = trajSelect.value;
  const t = DATA.trajectories[tid];
  _currentTraj = t;
  document.getElementById('traj-info').textContent = `T = ${t.T} steps`;

  const wins = getActiveWindows();

  // legend strip (single, top of plots)
  const ls = document.getElementById('legend-strip');
  ls.innerHTML = '';
  if (wins.length === 0) {
    ls.textContent = '(no windows selected — pick at least one in the sidebar)';
  } else {
    ls.appendChild(document.createTextNode('Windows: '));
    wins.forEach((w, i) => {
      const item = document.createElement('span');
      item.className = 'legend-item';
      item.innerHTML = `<span class="legend-swatch" style="background:${colorFor(i, wins.length)}"></span>${w}`;
      ls.appendChild(item);
    });
  }

  // initial cursor at step 0
  const initialShapes = cursorShape(0);

  // 8 plots
  for (const desc of DATA.descriptors) {
    for (const src of ['a', 't']) {
      const plotId = `plot-${src}-${desc}`;
      const traces = wins.map((w, i) => ({
        x: t.steps,
        y: t.factors[`f1b_${src}_${desc}__${w}`] || [],
        mode: 'lines+markers',
        name: w,
        line: { color: colorFor(i, wins.length), width: 2 },
        marker: { size: 4 },
        connectgaps: false,
      }));
      Plotly.react(
        plotId,
        traces,
        {
          title: { text: `${src.toUpperCase()} · ${desc}`, font: { size: 10 }, y: 0.97 },
          margin: { t: 18, b: 22, l: 36, r: 6 },
          showlegend: false,
          xaxis: { tickfont: { size: 8 }, automargin: false },
          yaxis: { tickfont: { size: 8 }, automargin: false },
          hovermode: 'closest',
          shapes: initialShapes,
        },
        { responsive: true, displayModeBar: false },
      );
      // wire plot-click → video seek (idempotent: Plotly.react preserves
      // existing listeners only across same div; we re-bind every render
      // to be safe — duplicate handlers are harmless because we always
      // overwrite the same global state)
      const div = document.getElementById(plotId);
      div.removeAllListeners?.('plotly_click');
      div.on('plotly_click', (ev) => {
        if (!ev || !ev.points || ev.points.length === 0) return;
        const x = ev.points[0].x;
        const traj = _currentTraj;
        if (!traj || traj.T < 2) return;
        const pct = Math.max(0, Math.min(1, x / (traj.T - 1)));
        const v = document.getElementById('trajvideo');
        if (!v.duration || !isFinite(v.duration)) {
          // metadata not loaded yet — at least move the cursor visually
          setCursor(x);
          return;
        }
        v.currentTime = pct * v.duration;
        // setCursor will fire via the video's timeupdate event
      });
    }
  }
}

// ---- video element + sync ----
const videoEl = document.getElementById('trajvideo');

function loadVideoForTraj(tid) {
  // The http.server is rooted at exp/common/analysis/; videos symlinked
  // to ./videos/ point at libero_spatial_replay_videos_fixed/.
  const url = `videos/${encodeURIComponent(tid)}.mp4`;
  videoEl.src = url;
  videoEl.load();
}

// Cursor sync — `timeupdate` fires only ~4 Hz so the cursor would jump
// in 250 ms chunks. Drive a `requestAnimationFrame` loop while the video
// is playing so the cursor advances at display refresh rate (~60 Hz).
function updateCursorFromVideo() {
  const traj = _currentTraj;
  if (!traj || traj.T < 2) return;
  if (!videoEl.duration || !isFinite(videoEl.duration)) return;
  const pct = videoEl.currentTime / videoEl.duration;
  setCursor(pct * (traj.T - 1));
}

let _rafHandle = null;
function startCursorLoop() {
  if (_rafHandle !== null) return;
  function tick() {
    updateCursorFromVideo();
    if (!videoEl.paused && !videoEl.ended) {
      _rafHandle = requestAnimationFrame(tick);
    } else {
      _rafHandle = null;
    }
  }
  _rafHandle = requestAnimationFrame(tick);
}

videoEl.addEventListener('play', startCursorLoop);
videoEl.addEventListener('pause', updateCursorFromVideo);
videoEl.addEventListener('seeked', updateCursorFromVideo);
videoEl.addEventListener('timeupdate', updateCursorFromVideo);  // safety net

videoEl.addEventListener('error', () => {
  // Show a discreet hint inside the panel without breaking the rest.
  const panel = document.getElementById('video-panel');
  if (!panel.querySelector('.video-missing')) {
    const m = document.createElement('div');
    m.className = 'video-missing';
    m.textContent = `(no video found for ${trajSelect.value})`;
    panel.appendChild(m);
  }
});

videoEl.addEventListener('loadedmetadata', () => {
  // Clear any prior "missing" notice.
  const panel = document.getElementById('video-panel');
  panel.querySelectorAll('.video-missing').forEach(n => n.remove());
});

// switch trajectory → reload video too
trajSelect.removeEventListener('change', render);
trajSelect.addEventListener('change', () => {
  loadVideoForTraj(trajSelect.value);
  render();
});

// keep plots sized to their grid cell when the viewport changes
window.addEventListener('resize', () => {
  PLOT_IDS.forEach(id => {
    const div = document.getElementById(id);
    if (div && div._fullLayout) Plotly.Plots.resize(div);
  });
});

// ---- Picture-in-picture ----
// HTML5 PiP API: pops the <video> into a floating always-on-top window
// (browser-native; not a re-render). The cursor sync continues to work
// because the same <video> element keeps firing play/pause/timeupdate
// events from inside the PiP window, and our rAF loop reads the same
// `videoEl.currentTime`.
const pipBtn = document.getElementById('pip-btn');
if (!('pictureInPictureEnabled' in document) || !document.pictureInPictureEnabled) {
  pipBtn.disabled = true;
  pipBtn.title = 'Picture-in-picture not supported by this browser';
}
pipBtn.addEventListener('click', async () => {
  try {
    if (document.pictureInPictureElement) {
      await document.exitPictureInPicture();
    } else {
      await videoEl.requestPictureInPicture();
    }
  } catch (e) {
    console.error('PiP toggle failed:', e);
  }
});
videoEl.addEventListener('enterpictureinpicture', () => {
  pipBtn.classList.add('active');
  pipBtn.textContent = '⧉ Exit PiP';
});
videoEl.addEventListener('leavepictureinpicture', () => {
  pipBtn.classList.remove('active');
  pipBtn.textContent = '⧉ PiP';
});

// initial load
loadVideoForTraj(trajSelect.value);
render();
</script>
</body>
</html>
"""


def serve() -> None:
    os.chdir(OUT.parent)
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/{OUT.name}"
        print(f"Serving at {url}")
        print("Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


def main() -> None:
    print(f"Loading {PKL} ...")
    data = load_data()
    print(f"Loaded {len(data)} trajectories "
          f"(min T={min(d['T'] for d in data.values())}, "
          f"max T={max(d['T'] for d in data.values())})")
    default_traj = max(data, key=lambda t: data[t]["T"])
    print(f"Default trajectory: {default_traj}")
    build_html(data, default_traj)
    if "--no-serve" in sys.argv:
        return
    serve()


if __name__ == "__main__":
    main()
