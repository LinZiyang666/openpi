"""Summarize smoke/*.txt into a table (method x cell: err, AURC, grip, B0 err/AURC, fit s, ms/query)."""
import pathlib
import re
import sys

d = pathlib.Path(__file__).with_name("smoke")
rows = []
for p in sorted(d.glob("*__*.txt")):
    name, cell = p.stem.split("__")
    t = p.read_text()
    def g(pat, s=t):
        m = re.search(pat, s)
        return float(m.group(1)) if m else float("nan")
    mm = re.search(r"metrics \[" + re.escape(name) + r"\]\s+(.*)", t)
    mb = re.search(r"metrics \[B0_current\]\s+(.*)", t)
    fit = g(r"fit=([\d.]+)s")
    q = g(r"q=([\d.]+) ms/query")
    if not mm:
        rows.append((name, cell, "NO METRICS"))
        continue
    M, B = mm.group(1), mb.group(1) if mb else ""
    rows.append((name, cell, g(r"n=(\d+)", M), g(r"err_mean=([\d.]+)", M), g(r"aurc=([\d.]+)", M), g(r"grip_mis=([\d.]+)", M),
                 g(r"err_mean=([\d.]+)", B), g(r"aurc=([\d.]+)", B), g(r"grip_mis=([\d.]+)", B), fit, q,
                 "PASS" if "SMOKE PASS" in t else "FAIL"))
print(f"{'method':28s} {'cell':20s} {'n':>4s} {'err':>6s} {'aurc':>6s} {'grip':>6s} | {'B0err':>6s} {'B0aurc':>6s} {'B0grip':>6s} | {'fit_s':>6s} {'ms/q':>7s}")
for r in rows:
    if len(r) == 3:
        print(r)
        continue
    print(f"{r[0]:28s} {r[1]:20s} {int(r[2]):4d} {r[3]:6.3f} {r[4]:6.3f} {r[5]:6.3f} | {r[6]:6.3f} {r[7]:6.3f} {r[8]:6.3f} | {r[9]:6.2f} {r[10]:7.2f} {r[11]}")
