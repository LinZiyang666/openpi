"""Re-render the generated tables of A100_CLOSEDLOOP_RESULTS_1018.md (CPU)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
doc = HERE / 'A100_CLOSEDLOOP_RESULTS_1018.md'
subprocess.run([sys.executable, str(HERE / 'a100_closedloop_analysis_1018b.py')], check=True, capture_output=True)
tables = subprocess.run([sys.executable, str(HERE / 'a100_closedloop_report_tables_1018b.py')], check=True,
                        capture_output=True, text=True).stdout
s = doc.read_text()
head, rest = s.split('<!-- RESULTS -->', 1)
_, tail = rest.split('<!-- /RESULTS -->', 1)
body = ('\n## 4. 闭环配对结果（表由 `a100_closedloop_report_tables_1018b.py` 从 '
        '[analysis JSON](evidence/a100_closedloop_analysis_1018b.json) 生成）\n\n'
        'ΔSR = A100 − 正式参考（单位 pp）；b/c = 仅 A100 成功 / 仅正式成功（同卡行为 c1 / c2）；'
        'p 为 exact 双侧 McNemar；CI 为预注册保守 exact 95% 区间（各 cell Clopper-Pearson 的 Bonferroni 并），'
        '等价门槛 ±2 pp。N 列括号内是并入的块（基础 100 + 扩展块）；两次冷启动从不合并成更大的 N。\n\n' + tables + '\n')
doc.write_text(head + '<!-- RESULTS -->' + body + '<!-- /RESULTS -->' + tail)
print('updated')
