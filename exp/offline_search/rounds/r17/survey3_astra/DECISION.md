| 决定 | 内容 |
|---|---|
| **首选** | **[Fast-WAM](https://github.com/yuantianyuan01/FastWAM)，用 [LeRobot 的冻结视频专家路径](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/modeling_fastwam.py#L80)微调 MT50。** 已核实 LIBERO 权重、单帧推理及部分微调代码；优先用一张 H100 80 GB。 |
| **备选** | **[Light-WAM](https://github.com/L1ziang/Light-WAM)**：训练更轻；仅在允许用 **VAE 潜变量作浅层检索键**、且检索质量验证通过时替补。深层视觉特征作键则缓存不划算。 |
| 证据边界 | **尚不能宣布任一模型五项无条件通过。** Fast-WAM 的 MT50 收敛/单卡吞吐，以及编译、减步后的缓存收益未实测。✓ 已证实；✗ 不满足；△ 有条件/估算；？未证实，不计通过。详见 [EVIDENCE.md](EVIDENCE.md)。 |

| 候选 | ① LIBERO 权重＋SR | ② MT50 便利性 / GPU 小时 | ③ 非 VLA、不同家族 | ④ 当前观测、无历史 | ⑤ 特征之后的缓存收益 |
|---|---|---|---|---|---|
| **Fast-WAM** | ✓ `.pt`＋LeRobot safetensors；迁移版均值 **96.4%**；原论文 97.6% | △ **约 13–20 H100 h** 起跑；30k 更新约 18–29 h；冻结约 5B，训练约 1B | ✓ Wan 视频 WAM；独立冻结 T5 文本编码 | ✓ `predict_action_chunk`；调用内 KV 每次重建 | ✓ 已发表配置约 **8.46×**；△ 最新编译/低 NFE |
| **Light-WAM** | ✓ 四套 `.pt`；**97.2%** | △ **约 16–24 H100 h**，在线 VAE；约 0.44B 可训；需适配 v3 数据 | ✓ 视频 WAM＋直接动作解码 | ✓ 单张图＋当前状态 | △ 浅 VAE 键 **5.36×**；✗ 深层键 **1.03×** |
| UVA | ✓ `libero10.ckpt`；Long **93%** | ？公开训练；未证明 ≤30 h，先因④淘汰 | ✓ 联合视频/动作生成 | **✗ 16 帧观测** | ？整次约 230 ms，未取得可用键后分项 |
| UWM | **✗ 仅核实 `libero_90.pt`**；79% 是另行逐任务微调的五任务结果 | ？Zarr 转换；未估，①④淘汰 | ✓ 联合视频/动作扩散 | **✗ 2 帧** | ？ |
| VPP | ？发布表仅 SVD/DP-CALVIN，未核实 LIBERO 权重 | ？8 卡参考配方；先因①淘汰 | ✓ 视频预测＋动作策略 | ？ | ？ |
| PAD | ？列到 Bridge、MetaWorld 权重，未列到 LIBERO | ？原训练 4×A800；先因①淘汰 | ✓ 联合图像/动作 DiT | ？ | ？ |
| Seer | ✓ Drive `33.pth/36.pth/38.pth`；Long **87.7%**，top-3 平均 | ？251M 冻结＋65M 可训；先因④淘汰 | ✓ 预测逆动力学 | **✗ 7 帧队列** | ？ |
| MoDE | ✓ `model_cleaned.safetensors`；Long **92%** | △ 原 LIBERO 4×A6000×2h＝8 GPUh；非 MT50 预算 | **✗ 仍是图像条件扩散 Transformer/MoE** | △ 单帧；绕过 action queue，任务专家缓存须正确重置 | ？ |
| Octo | △ 社区 LIBERO 文件存在；未核实对应 SR | ？先因①④淘汰 | ✗ 图像 Transformer＋扩散头 | ✗ 已查配置 window=2 | ？ |
| MDT | ？官方 CALVIN；查到的 LIBERO HF 仓仅 `.gitattributes` | ？先因①③淘汰 | ✗ 图像条件扩散 Transformer | ？ | ？ |
| GE-Act | ？V1 为基座/CALVIN；V2 未核实 LIBERO 权重＋完整训练入口 | ？先因①淘汰 | ✓ 视频模型路线 | ？ | ？ |
| RIFT | ✓ `rift_step021700.pt`；**98.8%** | ？现成入口全训约6B、8卡示例；无冻结专家/LoRA配方，预算未核实 | ✓ 视频 WAM | ✓ 当前帧生成 anticipation tokens | ？247.9 ms/A800，缺检索键后分项 |

| 单卡 MT50 配方（估算，非已跑通实验） | 首选 Fast-WAM | 备选 Light-WAM |
|---|---|---|
| 可训练部分 | LeRobot `freeze_video_expert=true`；动作专家＋重置的 4-D 动作输入/输出层、状态投影；T5/VAE 冻结；未确认专用 LoRA 配方 | 官方冻结 Wan1.3B，训练 LoRA-r64、adapter、视频头、StateFusion 动作专家：约 0.44B |
| 数据/更新 | 单相机 224²，state/action=4，horizon=32；**20k 更新×有效 batch16＝320k 窗口**；micro4×累积4 ⇒ LeRobot `steps=80000` | 同样单相机/4-D；**30k 更新×batch16＝480k 窗口**；v3→仓内 v2.1 loader 适配；在线 VAE，文本预编码 |
| 吞吐依据 | 公开 4H100 全训 **31.6 sample/s** → 每卡线性参考 **7.9**；规划按 **5–8 sample/s**，不是单卡实测 | 公开在线 VAE 消融 4H100 **35.7 sample/s** → 每卡参考 **8.925**；规划按 **6–9 sample/s**；缓存＋下采样可更快 |
| 小时算式 | 320k÷(5–8)÷3600＝**11.1–17.8 h**＋1–2 h 预留≈**13–20 h**；30k 更新＝**18–29 h** | 480k÷(6–9)÷3600＝**14.8–22.2 h**＋1–2 h 预留≈**16–24 h**；不依赖未知视频预计算耗时 |
| 硬件/限度 | **选 H100 80 GB**，BF16＋checkpointing＋micro4；约1B优化器可显著省显存，但48GB卡训练峰值/速度未核实；`lambda_video=0` **不会跳过视频前向** | H100：在线消融48.6 GiB/卡（micro8×累积2）；缓存版43.1 GiB。4090 48GB 需更小 batch，峰值/吞吐未核实 |

| 缓存核算：语言已缓存；δ＝新增池化/检索/修正开销 | 全调用 ms | 键提取＋其余保留开销 ms | full:hit，δ=0 上限 | 每命中省 ms |
|---|---:|---:|---:|---:|
| π0.5，用户 4090 基准 | 68 | 10.30 | **6.60×** | **57.70** |
| Fast-WAM，公开 4090 48GB、RoboTwin 输入；VAE 11.3＋视频专家36.0＋动作356.8 | 404.62 | 47.82＋δ | **8.46×** | **356.8−δ** |
| Fast-WAM，独立 H100 测试、LIBERO、10步；VAE5.5＋prefill30.3＋状态0.1＋动作310 | 345.9 | 35.9＋δ | **9.64×** | **310−δ** |
| Fast-WAM，4090 编译版仓库声称值 | 110（含 T5） | **未知** | **未知** | **未知** |
| Fast-WAM，1步敏感性：仅将上面动作项÷10，**不是测量** | 83.50 | 47.82＋δ | **1.75×** | **35.68−δ** |
| Light-WAM，4090 深层视觉特征键 | 72.03 | 69.93＋δ | **1.03×** | **2.1−δ** |
| Light-WAM，拟用浅 VAE 键，尚无检索质量验证 | 72.03 | 13.43＋δ | **5.36×** | **58.6−δ** |

| 落地判断 | 内容 |
|---|---|
| Fast-WAM 的验证门槛 | 预算内先验证 MT50 少量更新吞吐、显存和闭环可学性；继续到20k更新；若吞吐≥5 sample/s且有收益，再延长到30k。预算保证运行量，**不保证 MT50 收敛**。 |
| 缓存门槛 | 比较可接受成功率下的 **compiled＋NFE1/2/10**，不能只展示10步。深层键取当前图像 prefill 的池化特征，按 task 分库并约束当前 proprio；检索命中率/误差另测。 |
| 关键链接 | [首选权重](https://huggingface.co/ZibinDong/fastwam_libero_uncond_2cam224/tree/main) · [官方训练代码](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/fastwam/modeling_fastwam.py#L80) · [备选权重](https://huggingface.co/l1ziang/lightwam-checkpoints/tree/main) · [公开分项/训练效率](https://arxiv.org/html/2606.08242v1#S4.SS4) · [减步报告](https://github.com/yuantianyuan01/FastWAM/issues/73) |
