# DP TF32-off 残余 cache 漂移：CPU 定位（2026-10-18）

**最早可观测差异是 retrieval distance，随后改变连续混合权重；51个超预算请求均由这条路径解释。不是这些请求的corrector增量；动作后处理只留下float32舍入量级残差。TF32-off仍未恢复DP部署等价。**

分析覆盖原双卡TF32-off比较中全部51个硬预算失败请求（35 head、16 tail），每请求八份记录：两GPU×两冷启动×两并发round，共408条。16个tail绑定同一client前一head的权重，按cursor=5取实际返回切片；不把tail当新的cache决策。每份response/数组与fit SHA校验；使用冻结runtime的原生 `_kernel_w`，不重拟合、不运行模型或GPU。工具见 [dp_cache_localise.py:20](dp_cache_localise.py:20)、[head/tail及混合回放:40](dp_cache_localise.py:40)。

| 连续环节 / 范围 | 结果 | 判断 |
|---|---:|---|
| donor行及顺序 | 408条在各自逻辑请求内一致 | 离散top1未翻转不代表连续动作相同 |
| 已记录donor距离，51个最坏action跨卡对 | 最大差0.0006828308105 | 最早已记录的差异点 |
| 距离→核权重，CPU逐份重算 | max-abs = **0** | 核函数可精确重现记录权重 |
| 连续权重跨卡差（各请求16种配对） | 最大0.000634983182 | 即使同donor，混合比例也不同 |
| corrector | 全部decision_index=0，增量norm=0 | 首决策未执行corrector；tail复用同一head |
| float32 weighted blend→DP内部normalizer逆变换→外部quantile逆变换 | 对wire实测动作最大残差 **2.384185791×10⁻⁷** | downstream CPU回放解释每条动作 |
| 两卡动作差的重构残差 | 最大 **2.384185791×10⁻⁷** | 相对0.000610232353实测最大差约0.039% |

最坏请求为cache client6/delivery0。所选最坏跨卡对的距离前3项：4090 `[45.96380997,45.97111893,45.99303436]`，H100 `[45.96380997,45.97145844,45.99371719]`。首个距离相同，但相对第8距离从0.0910148621变为0.0913543701；原生核 `exp(-((d-d0)/(d7-d0))²)` 因此给不同混合权重。实际stage1 batch覆盖B1…B8。不能只核对最近邻ID或nearest distance就宣称cache一致。

冻结代码依据（runtime identity `5927a753…f1fa1e`）：`exp/r17_integration/common/layers/recipe.py:245` 的 `_retrieve`，`:249–251` 的首步early距离平方矩阵式，`:268–273` 的核/归一化/混合，`:302` 的corrector仅step>0；同runtime `common/layers/math.py:73` 的核函数；`models/dp/normalization.py:34` 的两级反归一化，`exp/mw_finetune/dp/model.py:107` 的内部scale/offset。上述路径均相对 `/data/mw_finetune/qualification/formal_runtime_5927a753312bf5981cba81a2e389edea914bde0580523b0dc3c8ea8d58f1fa1e/`。CPU回放的normalizer参数只读取checkpoint两个7元素tensor和封存norm stats，未读取完整权重。

## 归因边界与下一步

已定位 **首个query的 early metric 距离→权重→cache动作**；尚未定位距离差来自stage1 feature的GPU/batch漂移，还是其后的CPU投影/BLAS舍入。记录不含query feature、projected x、y、d2，所以不能反推其中某个CUDA kernel为根因，也不能声称CPU重算权重就是已验证的修复。这里corrector为零的结论只覆盖51个失败请求，不排除其他查询的corrector漂移。

下一次DP诊断只需追加首head的stage1 feature、projected x、early y及原始d2；固定同一query feature在CPU分别回放原float32距离与float64距离，再比较是否是距离平方中的抵消放大。保留正式并发/graph/recipe，勿通过B1或事后放宽预算掩盖问题。未执行该GPU诊断。纯动作与cache的原硬预算、逐请求floor和闭环±2pp标准仍见 [DP_BOTH_TF32OFF_1018.md](DP_BOTH_TF32OFF_1018.md)。

证据：[完整408条回放](evidence/dp_cache_localisation_1018.json)、[摘要](evidence/dp_cache_localisation_summary_1018.json)、[normalizer](evidence/dp_cache_normalization.json)、[CPU执行日志](evidence/dp_cache_localise_run_1018.log)。完整证据SHA256 `9cd1dd119260322e0c5319a02e3ab3b33e83161bb36e82047e9a0afb58a4b7a1`；原比较SHA `727947cc…ac2fd`；fit NPZ SHA `b8bb9bc2…4f8`。计算在存放四次raw结果的H100上以 `CUDA_VISIBLE_DEVICES=`、`OPENBLAS_NUM_THREADS=1` 执行CPU回放，仅写研究证据，再用只读tether exec取回312,696 bytes JSON；无GPU/model/worker启动、无sudo。
