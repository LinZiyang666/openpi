"""Assemble the authored report and its numerical appendix; no external writes."""
from pathlib import Path

out = Path(__file__).resolve().parent
body = (out / "REPORT_body.md").read_text()
tables = (out / "MEASUREMENTS.md").read_text()
tables = tables.replace("# R4-A generated measurements", "## 5. Numerical appendix", 1)
tables = tables.replace("\n## ", "\n### ")
(out / "REPORT.md").write_text(body.rstrip() + "\n\n" + tables.rstrip() + "\n")
print(out / "REPORT.md")
