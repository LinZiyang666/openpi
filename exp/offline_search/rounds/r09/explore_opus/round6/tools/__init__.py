from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)
