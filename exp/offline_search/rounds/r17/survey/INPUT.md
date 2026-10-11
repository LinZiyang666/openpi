# R17 选型输入（2026-10-08 19:4x）

## owner 的要求（原话要点）
- 把整套系统（检索缓存 + 无进展守卫 + 库内修正器 + 开销旋钮 + 逐任务下降；有迭代去噪的再加单纯减步；有动作块的加扩大执行段）迁移到其他机器人策略上。"理论上有 vision head 的都可以把我们的系统移植过去（没有的也可以试一下？）"，包括非 VLA 策略。可参考单纯减步的经验。
- 模型：最好有 LIBERO 检查点。
- 另选一个 benchmark：比较新、影响力大、适合我们这种系统；所有或大多数候选模型都要支持，**π0.5 必须支持**。
- **benchmark 不能太重**（"太重不利于我们快速实验"）：我们一次实验几千到上万局，worker 是 timan107 上约 64 个 CPU 进程（MuJoCo 类、EGL/OSMesa 离屏渲染），服务端是单张 4090（48 GB）或 h100。

## 系统要点（详见 rounds/r16/REPORT.md、rounds/r15/REPORT.md）
- 检索键 = 策略自己视觉塔的特征（两路相机 16×2048 空间池化 → PCA64 + 状态 8 → 按任务马氏距离），top-16 核加权合成，10 步提交。
- 守卫：top-1 库帧进度停滞触发大模型调用；修正器：RFF + 岭回归修正缓存动作块；旋钮：定期调用；第 5 层：逐任务档位下降。
- R16：大模型调用改 k = 1/2 步去噪，π0.5 / GR00T 在 LIBERO 上与满步持平，完整系统开销 .17–.19 对满步纯推理 .51。

## 内部历史调研（Sonnet，只读本仓库日志；引用都在原日志）
- 跨模型只做过**单纯减步**，**从未把缓存移植到 π0.5 / GR00T 以外的模型**。
- 减步结果：
  - π0.5 LIBERO k=1 近无损；π0.5 RoboCasa365（13 原子任务）k=1 掉 16.6 个点、k≥5 持平；GR00T RoboCasa365 k=1 0.572 对 4 步 0.680。
  - Cosmos Policy 2B（EDM 视频扩散）LIBERO spatial k=1 0.974 对 0.986；RoboCasa-2024 k=1 0.6725（官方 0.671）。LIBERO / RoboCasa 检查点已从本机删除，重下需 NVIDIA 许可 + HF token。
  - X-WAM（Wan2.2-TI2V-5B 世界-动作模型）RoboCasa-2024 1 步无损（0.79）；RoboTwin 2.0 k=1 掉 4.4 个点（集中在 7 个任务）。每次查询 0.22–0.76 s。
  - Diffusion Policy（ε-DDPM，robomimic 5 任务）：DDIM-10 0.88、DDIM-2 0.60、DDIM-1 0.03；ε 头 k≤2 崩，x₀ 头 1 步近无损。无 LIBERO 检查点。
  - π0.5 MetaWorld MT50（RLinf SFT 检查点）：满步 0.588、k=2 0.604、k=1 0.537。
- 已做过：SmolVLA、ACT 蒸馏后当 π0.5 缓存的旁路执行器（不是用它们自己的特征建库）。
- 本机现成资产：π0.5 检查点（LIBERO、RoboCasa、MetaWorld），GR00T N1.5（LIBERO spatial/10、RoboCasa），X-WAM（RoboCasa / RoboTwin SFT），DP robomimic；环境 libero_sim、metaworld_sim、gr00t_n15_venv、lerobot_venv、RoboCasa365（在 Isaac-GR00T third_party）、RoboCasa-2024、RoboTwin（micromamba）。
- 教训：RoboTwin 很重——SAPIEN + CuRobo，CPU 瓶颈，7 个客户端约 160 局/小时，一轮 50 任务 × 20 局约 15 小时，结果不可逐位复现，OIDN 降噪 OOM 静默失败；timan107 上慢约 10 倍。

## 网上调研（Sonnet，2026-10-08；[UV] = 未核实）
- 模型候选：π0.5（官方 pi05_libero 四套）、π0 / π0-FAST（无官方 LIBERO）、GR00T N1.7 / N1.6（官方 GR00T-N1.7-LIBERO，独立复现 LIBERO-10 91.2%）、X-VLA-0.9B（2toINF/X-VLA-Libero 98.1 平均；另有 RoboTwin2、Simpler、CALVIN、VLABench 检查点；Florence-2 系编码器，flow 10 步，块约 30）、SmolVLA（HuggingFaceVLA/smolvla_libero）、FLOWER VLA（Florence-2 DaViT，4 步 flow，LIBERO 含 90、CALVIN；每套检查点是否公开未核实）、VLA-Adapter（0.5B，四套 LIBERO 检查点，非扩散头）、OpenVLA-OFT（7B，无迭代头，每套检查点）、InternVLA-M1（Qwen2.5-VL，每套检查点）、EO-1、RDT-1B（非 VLA 扩散 transformer，SigLIP + T5，块 64，无 LIBERO，RoboTwin 强）、MDT（非 VLA，CALVIN）、DP / ACT / VQ-BeT（LeRobot，LIBERO-90 只有二手数字，无官方 LIBERO 检查点）。2026 新的：Xiaomi-Robotics-0、VLAFlow 等未核实。
- benchmark 候选：
  - RoboCasa365（ICLR 2026，有排行榜：π0.5 16.9、GR00T N1.5 23.9、N1.6 21.9、π0 14.8、DP 6.1；600+ 小时示范；MuJoCo，单局较重 [UV]）。
  - RoboTwin 2.0（影响大，X-VLA / RDT / π0 / DP 都有；π0.5 只有第三方数字；很重，见上）。
  - LIBERO-Plus（CVPR 2026，7 种扰动，同 LIBERO 仿真与示范；π0.5 只有第三方数字；官方代码非语言扰动的 prompt 构造有问题）。
  - LIBERO-PRO（README 表含 π0.5，扰动后各模型都从 >0.9 崩到约 0）。
  - VLABench（LeRobot π0.5 base 42、X-VLA 51.1，导入行非官方）、SimplerEnv（无 π0.5 数字 [UV]）、BEHAVIOR-1K（π0.5 前两名，但 OmniGibson 很重）、CALVIN / ManiSkill3（无 π0.5）。
- 网上调研的建议（未考虑"不能太重"）：模型 GR00T N1.7、X-VLA、FLOWER、RDT-1B；benchmark RoboCasa365，其次 RoboTwin 2.0；LIBERO-Plus / PRO 作廉价附加。

## 网上调研补充：按"不能太重"重排（Sonnet 第二轮，2026-10-08 19:5x；单局耗时全是估计，未实测）
- benchmark 轻重排序：
  1. LIBERO-Plus（CVPR 2026）/ LIBERO-PRO：同 LIBERO 仿真栈，单局成本与现在相同，不新增依赖；π0.5 只有第三方数字（IndustrialVLA-Bench 85.3 robust avg，DFM-VLA 75.7）；Flex-π 指出 LIBERO-Plus 官方代码 prompt 构造有 bug，需固定一套协议。
  2. Meta-World MT50：MuJoCo，轻；π0 / π0.5 只有社区重训（RLinf、πRL），无官方检查点。
  3. RoboCasa / RoboCasa365：MuJoCo，约 10 GB 资产，单局比 LIBERO 长；π0.5 / π0 / GR00T N1.5 / N1.6 / DP 都在排行榜上。
  4. CALVIN：PyBullet CPU，轻；无已核实的 π0.5 结果。
  5. VLABench：MuJoCo / dm_control（未核实）；π0.5 只有导入数字 42。
  6. SimplerEnv：SAPIEN 必须 GPU 渲染；7. RoboTwin 2.0：SAPIEN + CuRobo，重；8. ManiSkill3：GPU 并行仿真；9. BEHAVIOR：Isaac / OmniGibson，最重。
- 模型建议更新：X-VLA、GR00T N1.7/N1.6、OpenVLA-OFT（7B，无迭代头，缓存省得最多，只能用扩大执行段）、FLOWER（4 步 flow，<3 GB，LIBERO 含 90）；非 VLA 对照用 LeRobot 的 Diffusion Policy 或 ACT（需自训 LIBERO，第三方 DP LIBERO 数字 78.3/92.5/68.3/50.5）。
