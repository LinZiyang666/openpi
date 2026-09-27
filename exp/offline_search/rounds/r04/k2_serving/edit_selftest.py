from pathlib import Path
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/selftest.py');s=p.read_text()
s=s.replace('def main(argv=None):', Path('exp/offline_search/rounds/r04/k2_serving/dev/blind_selftest_snippet.txt').read_text()+'def main(argv=None):')
s=s.replace('    a = ap.parse_args(argv)\n','    ap.add_argument("--blind", action="store_true", help="run the interleaved R4 blind serving test")\n    a = ap.parse_args(argv)\n    if a.blind:\n        return blind_main(a)\n')
p.write_text(s)
