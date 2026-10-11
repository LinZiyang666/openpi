"""Build an auditable report from locally saved public metadata; no network/mutations outside O."""
from pathlib import Path
import json, re, collections, csv

O = Path(__file__).resolve().parent
R = O / 'raw'
def read(p): return json.loads(p.read_text())
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] + ['| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in row)+' |' for row in rows])+'\n'
def meta(mid): return read(R/'hf'/mid.replace('/','__')/'metadata.json')
def link(mid): return f'[{mid}](https://huggingface.co/{mid}/tree/main)'
def files(mid):
    m=meta(mid)
    return table(['文件（完整仓库 listing）','bytes'], [(f"`{x['rfilename']}`",x.get('size','?')) for x in m['siblings']])
def snap(mid):
    m=meta(mid)
    return f"{link(mid)} · created `{m.get('createdAt')}` · revision `{m.get('sha')}` · gated `{m.get('gated')}` · card license `{m.get('cardData',{}).get('license','未声明')}` · [API 快照](raw/hf/{mid.replace('/','__')}/metadata.json)\n\n"

terms=['metaworld','meta-world','meta_world','mt50','mt10','libero']
catalog={}; queries=collections.defaultdict(list)
for term in terms:
    for m in read(R/f'hf_search_{term}.json')['models']:
        catalog[m['id']]=m; queries[m['id']].append(term)
with (O/'HF_CATALOG.tsv').open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['id','author','query_hits','createdAt','lastModified'])
    for mid,m in sorted(catalog.items()):w.writerow([mid,mid.split('/')[0],','.join(queries[mid]),m.get('createdAt',''),m.get('lastModified','')])
mw=[mid for mid in sorted(catalog) if re.search(r'metaworld|meta[-_]world|mt50|mt10',mid,re.I) and not any(s in mid.lower() for s in ['metaclip','temporalgames/','hansbug/','hayatohongo/','muapi/'])]
def family(mid):
    s=mid.lower()
    if '/dp-' in s or '/diffusion-' in s:return 'Diffusion Policy'
    if 'cosmos' in s:return 'Cosmos Policy'
    if 'crt_' in s:return 'CRT（修复器）'
    if 'evo' in s:return 'Evo / InternVL VLA'
    if 'smolvla' in s or 'svla_' in s or 'rskill-smol' in s:return 'SmolVLA / 派生'
    if 'openvla' in s:return 'OpenVLA'
    if 'xvla' in s or 'x-vla' in s:return 'X-VLA'
    if 'gr00t' in s:return 'GR00T'
    if 'pi0' in s or 'alam_' in s:return 'π 家族 / 派生'
    if 'act' in s or mid.startswith('Muhammad241198/'):return 'ACT'
    if 'spur' in s or 'qwen' in s or 'tinyvla' in s or 'vla-' in s:return '其他 VLA'
    if mid.startswith('ksb21st/'):return 'π / Orbax 格式；无新增非 VLA 证据'
    if mid.startswith('Master210/'):return '仅 MW，未建立 LIBERO 家族对应'
    return '未确立双基准完整策略'
with (O/'METAWORLD_AUDIT.tsv').open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['id','family_screen','full_file_metadata_saved'])
    for mid in mw:w.writerow([mid,family(mid),(R/'hf'/mid.replace('/','__')/'metadata.json').exists()])

out=[]
out.append(table(['证据账本','口径'],[
    ['日期 / 独立性','2026-10-08；仅本输出目录写入。未读其他 survey、prompts、ops 目录；参考系统背景仅 r16/REPORT.md。'],
    ['最终结论','0 个完成全部认证；DP / brandonyang 为条件候选。见 DECISION.md。'],
    ['查询完整度',f'六个指定查询按 HTTP Link rel=next 翻至结束，去重 {len(catalog)} 仓库；不是只取 LIBERO 前1000条。'],
    ['范围限制','公开可访问、可索引的来源快照；私有/未索引/异名或失效链接可能漏检。缺证不等于不存在。'],
    ['安全下载','不下载完整策略权重；DP 仅读两份3.6KB metadata和两个128KiB tensor header；BAKU ZIP 仅读64KiB尾部。'],
    ['测量限制','CPU 随机权重结构代理，不是 checkpoint 成功率复现，也不能外推 GPU 毫秒数。'],
]))
out.append('**HF 枚举：实际调用及翻页。** [脚本](enumerate_hf.py)、[去重目录（5014行）](HF_CATALOG.tsv)、[MetaWorld 名称筛选及家族归并](METAWORLD_AUDIT.tsv)。\n')
rows=[]
for term in terms:
    d=read(R/f'hf_search_{term}.json')
    rows.append([f'`{term}`',len(d['models']),len(d['pages']),'+'.join(str(x['count']) for x in d['pages']),f'[首请求](https://huggingface.co/api/models?search={term}&limit=1000&full=true)',f'[全部响应/游标](raw/hf_search_{term}.json)'])
out.append(table(['query','返回条数','页数','各页','API','本地快照'],rows))
out.append(f'`meta-world` / `meta_world` 搜索是宽松词匹配，大量结果是 MetaCLIP worldwide、文本模型或其他噪声。以 `metaworld|meta[-_]world|mt50|mt10` 名称筛选并剔除明显非机器人命中后保留 **{len(mw)}** 个名称线索；这个数仍不是合格模型数。每个线索的文件是否单独核查，已在 TSV 标明。HF 数据集仓库不算 model weights。\n')
out.append(table(['双边名称共同作者（16）','家族归并 / 后续核查'],[
    ['AIRLab-POLIMI','CRT；是真文件，但不是 action policy。'],['brandonyang','DP 是核心候选；另有 π0-FAST/openpi，同作者不等于同架构。'],['iFaz','DP；MW 空仓。'],['kunhsiang','ACT / inverse / feedback ACT；抽查完整双边权重。'],['MINT-SJTU','Evo-1、EVO-Depth、Evo-World、StarVLA；均需按 VLM 骨干排除。'],['RLinf','π / OpenVLA；DreamZero/Wan/OpenSora 的 LIBERO 文件不自动建立 MW 交集。'],['Mark-ZJTang','ALAM+π0 双边；不能把 action tokenizer 当完整非 VLA。'],['lerobot','SmolVLA 双边；FastWAM/lingbot/其他 LIBERO 文件未建立 MW 交集。'],['OpenRAL','rskill 封装；双边 SmolVLA 仍属同家族。'],['JunnDooChoi','π0.5/SmolVLA 的 UFOS/MLP 等 head 替换，骨干仍 VLA。'],['jadechoghari','SmolVLA 双边；X-VLA、tokenizer 不能和 SmolVLA 混拼。'],['msv6 / nikriz / tiantianx','SmolVLA 或 π 派生，合并而不重复计推荐。'],['zhongweihong','X-VLA 单仓双子目录，完整文件已核查。'],['Rhx11111','MW-256 空仓，LIBERO 名称也不构成同家族完整发布。'],
]))
out.append('除作者交集外，跨作者也按家族匹配：官方 Cosmos ↔ mmurray-ms；官方 GR00T ↔ namenu；BAKU/RT-1/ACT 在同一 OSF 压缩包。补充 HF query `baku, quest, cocos, canonical, vpp, GEVRM, DiffuView, DynaMo, flowpolicy, diffusion_metaworld` 的原始结果为 `raw/hf_extra_*.json`；这些是补充搜索，**不宣称都完整翻页**（如 quest 命中1000条有截断）。\n')
out.append('**公开网页检索及论文交叉。** 使用组合词 `"LIBERO" "MetaWorld" "diffusion" "checkpoints"`、`"LIBERO" "MetaWorld" "world model" weights`、`"LIBERO" "MetaWorld" "2026" "policy" non VLA`；对命中方法追踪官方论文→代码→model/Drive/OSF链接→文件清单。另查询 `VQ BeT pretrained LIBERO MetaWorld checkpoints github`、`PRISE pretrained checkpoints LIBERO MetaWorld github`、`SUREFlow checkpoint github`。依据以论文/作者仓库/HF 原始文件为准，不以聚合站摘要定论。\n')

out.append('<a id="dp"></a>\n\n**DP / brandonyang：实际文件、配置和推理链。**\n')
for mid in ['brandonyang/dp-libero-checkpoints','brandonyang/dp-metaworld-checkpoints']:
    out.extend([snap(mid),files(mid)])
out.append(table(['配置项目','LIBERO 30k','MetaWorld 85k'],[
    [k,read(R/'brandon_libero_config.json')['config']['model'][k],read(R/'brandon_metaworld_config.json')['config']['model'][k]]
    for k in ['action_dim','action_horizon','n_obs_steps','n_action_steps','horizon','lang_emb_dim','num_inference_steps','num_train_timesteps','beta_schedule','prediction_type','vision_backbone','crop_shape','obs_encoder_group_norm','eval_fixed_crop','n_layer','n_cond_layers','n_head','n_emb','causal_attn','time_as_cond','obs_as_cond']
]))
out.append('配置证据：[LIBERO 安全解析 JSON](raw/brandon_libero_config.json)、[MW 安全解析 JSON](raw/brandon_metaworld_config.json)、[解析脚本](probe_metadata.py)。`metadata.pt` 内不是只有训练步数，而是完整 model/data/optimizer 配置。用 `pickletools` 看结构后，仅允许 `flax.nnx.filterlib.Nothing` 映射为本地占位；未调用远端 checkpoint 中任意 Python 类、未加载模型参数。保存的 pickle disassembly 可复查。\n')
out.append('权重结构证据：[LIBERO safetensors header](raw/brandon_libero_safetensors_header.json)、[MW header](raw/brandon_metaworld_safetensors_header.json)。HTTP Range 仅读前131072 bytes；模型的 `_policy.model.*` 全部 tensor 名称/形状与上游 `TransformerForDiffusion` 一致，代理结果 `model_shape_mismatches={}`。两侧视觉都是 language-FiLM ResNet18，末端 spatial-softmax 32 keypoints 与64维线性层；2×64可作自己的视觉key。语言768维另由外部提供，发布目录没有 VLM/LLM 权重，也没说明具体文本 embedder。\n')
out.append(table(['代码审查','证据 / 限制'],[
    ['每次采样重新初始化','[上游 policy](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py)：conditional_sample 创建 randn、set_timesteps，并在本次调用内完成 denoising。'],
    ['观测窗口','同文件 predict_action 按 n_obs_steps 切取输入，assert 无 past_action；发布配置为1。随机采样本身不算观测历史。'],
    ['key 提取点','obs_encoder 后、model denoiser 前；[视觉编码器上游](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/vision/multi_image_obs_encoder.py)。checkpoint 多出 FiLM 自定义层，因此视觉代理是重建，不宣称逐算子相同。'],
    ['尚缺原样推理','HF作者主页关联 branyang02；其公开 openpi-eval/main及分支、openpi-trajectory、openpi-cuda、ricl_openpi_latent_action、vvla 的树/README未定位这个 DP wrapper。GitHub搜索 dp_libero_lang_v1、dp_metaworld、DiffusionTransformerHybridImagePolicy 作者范围也未找到。'],
    ['复现缺口','MW 为什么仍有 left_wrist_0_rgb、language embedding生成、裁剪/动作变换和 suite覆盖需专用代码或作者说明。配置可当前帧，不代表任意社区wrapper一定无历史。'],
    ['成功率','两 repo 无 README、eval metrics；所查关联代码也没给此 DP checkpoint 的 LIBERO/MW SR。不得借用原始 DP、BAKU、QueST 论文 baseline 数字。'],
    ['license / access','gated=false；两边无 LICENSE/model card。上游 DP MIT 不自动授权社区模型权重。'],
]))
out.append('作者溯源快照：`raw/brandon_profile.txt`、`raw/branyang02_repos.json`、`raw/branyang02__*_branches.json`、`raw/brandon_tree_*.json`、`raw/gh_search_dp_metaworld.json`、`raw/gh_search_diffusion_policy_brandon.json`。这些是“尚未找到 wrapper”的搜索范围，不是证明代码不存在。\n')

out.append('<a id="cost"></a>\n\n**DP 成本复核。** [可复跑 CPU 脚本](cpu_cost_probe.py)、[全部计时样本/参数与 FLOPs](raw/cpu_cost_proxy.json)、[运行日志](raw/cpu_probe.log)。\n')
out.append('```bash\ntaskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python exp/offline_search/rounds/r17/survey2_astra/cpu_cost_probe.py\n```\n')
j=read(R/'cpu_cost_proxy.json');v=j['vision_2cam']['median_ms'];q=j['lookup_5000x128']['median_ms']
out.append(table(['steps','vision ms','tail ms','full ms','hit ms','full/hit','saved ms'],[
    [s,f'{v:.6f}',f'{j[f"head_{s}"]["median_ms"]:.6f}',f'{v+j[f"head_{s}"]["median_ms"]:.6f}',f'{v+q:.6f}',f'{(v+j[f"head_{s}"]["median_ms"])/(v+q):.6f}',f'{j[f"head_{s}"]["median_ms"]-q:.6f}'] for s in [100,2,1]
]))
out.append('B=1 FP32；torch2.7.1、4 CPU intra-op /1 inter-op threads；无 GPU forward。随机权重，2次 warm-up；vision、1/2-step tail 各10样本；100-step tail 3样本（2527.285–2598.302ms）；lookup30样本。full 是 stage medians 相加，未作为端到端装载检查点计时。head 是上游12层 decoder+4层 condition Transformer、DDPMScheduler；视觉仅按 metadata/header 复原。100/2/1 steps 用相同参数，不测 action质量。\n')
out.append('参数量：vision28,300,096，head63,539,207；vision FLOPs1,197,098,368，单步 head profiler统计1,469,498,625。FLOPs 采用 multiply/add 各一次的 profiler convention；fused attention、归一化等可能漏计，因此是近似/下界，不写成完整硬件 FLOPs。128维 key 对5000条cache做内积+argmax+chunk取出，median0.058672ms。没有纠错器/guard成本。一般公式：full=E+D；hit=E+L+H；saved=D−L−H；ratio=(E+D)/(E+L+H)。这解释了为何先降步到1/2后缓存收益明显收窄。\n')

out.append('<a id="ifaz"></a>\n\n**iFaz：名称交集不是权重交集。**\n')
for mid in ['iFaz/diffusion-libero_goal-seed3-15000','iFaz/diffusion-metaworld_medium-seed42-half']:out.extend([snap(mid),files(mid)])

out.append('<a id="baku"></a>\n\n**BAKU / RT-1 / MT-ACT：官方 OSF 发布清单。** [官方 README](https://github.com/siddhanthaldar/BAKU)、[OSF project](https://osf.io/3x8v5/?view_only=fb8285f025e84d23a41a0eef683a7e6d)、[OSF API](https://api.osf.io/v2/nodes/3x8v5/files/osfstorage/?view_only=fb8285f025e84d23a41a0eef683a7e6d)。\n')
out.append('`weights.zip` 总长2,561,746,368 bytes。HTTP206只取 bytes2561680832–2561746367，共65536 bytes，解析 central directory而非下载2.56GB；[OSF metadata](raw/baku_osf.json)、[Range记录](raw/baku_zip_tail_meta.json)、[ZIP完整目录](raw/baku_zip_listing.json)。以下是其中完整的策略文件列表：\n')
out.append(table(['zip路径','解压后bytes','压缩bytes'],[(x['path'],x['size'],x['compressed']) for x in read(R/'baku_zip_listing.json') if x['path'].endswith('.pt')]))
out.append(table(['核查点','结果与原始来源'],[
    ['配置不在 ZIP 内','官方 [config_eval.yaml](https://github.com/siddhanthaldar/BAKU/blob/main/baku/cfgs/config_eval.yaml)、[LIBERO suite](https://github.com/siddhanthaldar/BAKU/blob/main/baku/cfgs/suite/libero.yaml)、[MW suite](https://github.com/siddhanthaldar/BAKU/blob/main/baku/cfgs/suite/metaworld.yaml) 和 agent yaml；已保存本地。'],
    ['BAKU骨干/head','ResNet18-FiLM，自身视觉feature→8层/4head/256dim GPT→2层确定性MLP；不是VLM。config_eval是deterministic，num_queries10；代码支持 diffusion 选项不证明发布 baku.pt 是 diffusion。'],
    ['BAKU历史区别','suite.history=false，所以 observation_buffer实际最大长度1；但 [act](https://github.com/siddhanthaldar/BAKU/blob/main/baku/agent/baku.py#L654) 写 self.all_time_actions，并对旧预测加权。R3失败来自 action temporal aggregation，不能误说默认10帧历史。'],
    ['RT-1','[Instructions](https://github.com/siddhanthaldar/BAKU/blob/main/Instructions.md#L89) 双基准都设置history=true、history_len6；普通 image-token Transformer，并非LLM。'],
    ['收益判断','BAKU tail 是少数当前观测tokens的一次小GPT+MLP；视觉是128×128 ResNet。未运行测量，故不填伪造毫秒/加速比。即使消除temporal aggregation，也缺后段占优的结构。ACT按owner现有排除条件处理。'],
    ['paper结果','[BAKU paper v2](https://arxiv.org/html/2406.07539v2) Table1：LIBERO90 / MW30，BAKU90/79、RT-1 16/65、MT-ACT54/13、VQ-BeT90/78（%）。VQ-BeT无ZIP权重。不是40task LIBERO平均或MT50。'],
    ['数值与release映射','作者发布该实验权重，但ZIP未含逐checkpoint评测日志；上述是论文方法数值，不是本轮复现。'],
    ['license/access','官方代码MIT；OSF匿名可读。未找到ZIP独立权重license说明。'],
]))

out.append('<a id="canonical"></a>\n\n**Canonical：有 release 声明，但发布文件未核实，且推理不合要求。** [代码](https://github.com/agentic-learning-ai-lab/canonical)、[所链HF](https://huggingface.co/agentic-learning-ai-lab/canonical)、[HTTP401记录](raw/hf_canonical.log)。不能把401直接说成“必须申请”或“仓库不存在”。\n')
canon=O/'code/agentic-learning-ai-lab__canonical'
cfgs=sorted(x for x in canon.rglob('*') if x.is_file() and x.suffix in ['.yaml','.yml'])
for f in cfgs:
    text=f.read_text()
    if any(k in text for k in ['obs_window','DiTFlow','goal_dim']):
        vals=[line.strip() for line in text.splitlines() if re.search(r'obs_window|action_chunk|n_action_steps|goal_dim|use_libero_goal|state_dim|num_inference_steps|num_steps|n_steps|depth|n_layer',line)]
        out.append(f'[{f.relative_to(canon)}]({f.relative_to(O)})：`'+ '; '.join(vals).replace('`','')+'`\n')
out.append('LIBERO：ResNet18→DiTFlowPolicy（dim224/depth4，50steps），obs_window10，goal_dim512/use_libero_goal=true，action chunk1。MW：ResNet18 spatial→DiTFlowSpatial（dim256/depth8，通常10steps、hard30steps），obs_window2，chunk4/execute2；state11由hand position4+velocity3+prev_action4组成。读取 [LIBERO train/eval](code/agentic-learning-ai-lab__canonical/libero_goal/train_policy.py)、[MW train/eval](code/agentic-learning-ai-lab__canonical/metaworld_bench/train_policy.py)、[policy实现](code/agentic-learning-ai-lab__canonical/models/policy.py)，不仅看模型名称。仓库声称的 `metaworld/encoder.pt` 等仅是文档示例，**不能当实际文件清单**。代码MIT，权重license/access未知。\n')

out.append('<a id="cocos"></a>\n\n**Cocos：最值得等待双边发布证明的新 flow 家族。** [论文](https://arxiv.org/abs/2505.11123)、[NeurIPS正式版本](https://papers.nips.cc/paper_files/paper/2025/file/ad6363efb7af02f7db13d087c7e649bd-Paper-Conference.pdf)、[官方代码](https://github.com/ZibinDong/cocos)、[OneDrive原链接](https://1drv.ms/u/c/ba682474b24f6989/EdyFZAAU47BPtZohTYDMHfwBeWB51Z8wSupl6z-iqfqBuw?e=kjBpzM)。\n')
out.append(table(['项目','核实'],[
    ['发布边界','README只给LIBERO实现和一个压缩包下载链接。OneDrive landing/SharePoint请求失败；shares API未得root文件表（404）。无可验证文件名、bytes/config，故R1不通过，不能说已证实双边发布。raw/cocos_onedrive_share.txt保留响应。'],
    ['默认LIBERO推理','[cocos.py](code/ZibinDong__cocos/cocos.py) To1、Ta16、execute8、sampling_steps20、Euler rectified flow；每次以最新双视角obs重算x1与condition，默认未启用autoencoder EMA更新，无跨调用KV。MW对应代码未找到。'],
    ['架构','DINOv2-base→patch features（可pool为key），T5-base任务embedding→约40M RDT/DiT head；12层/384dim/6heads；condition-conditioned source用小AE。T5独立文本encoder不等于VLM/LLM action backbone。'],
    ['key成本风险','公开推理 process_obs 通过AE的DINO编码；condition.forward又调用vit_model。需要把两个视觉路径/AE算入full与hit，不能只把小AE后的tensor视作免费key。20步head是否主导需实测。'],
    ['paper LIBERO','Cocos std0.2：Goal93.8±0.3、Spatial97.5±0.7、Object99.1±0.6、Long88.7±0.1，平均94.8（%）；见raw/cocos_paper.html。'],
    ['paper MW','30 tasks，74.8±0.9%，baseline59.5±3.8%。不是MT50/ML45；这一自选30任务是否符合owner标准也需单独确认。'],
    ['license/gating','所查代码树无LICENSE；OneDrive访问失败不能归类为已知gating。没有可独立认证的发布checkpoint SR。'],
]))

out.append('<a id="cosmos"></a>\n\n**Cosmos：仅补充事实，不改变 owner 已定范围。**\n')
for mid in ['nvidia/Cosmos-Policy-LIBERO-Predict2-2B','mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B']:out.extend([snap(mid),files(mid)])
out.append('官方LIBERO [model card](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)：Cosmos-Predict2-2B视频骨干，5 denoising steps、16×7 action；Spatial98.1/Object100/Goal98.2/Long97.6，平均98.5%。license是 **NVIDIA One-Way Noncommercial License (NSCLv1)**，不是Apache或通用开放商用许可。\n')
out.append('社区MW [model card](https://huggingface.co/mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B)：FlowDAgger artifact，仅assembly-v3；base约52%/25episodes，full-mode steering84–88%；后者不能归为base policy。要求gripper proprio `[g,-g]`。仓库无模型config；`dataset_statistics.json`只含归一化，不替代架构配置。card license为other、无明确许可文本。所查 [microsoft/FlowDAgger](https://github.com/microsoft/FlowDAgger) 公开main主要为π0.5 MW与GR00T LIBERO配方，未定位该Cosmos runner。两边匿名可列文件，但MW范围和配套代码均不足。\n')

out.append('<a id="other-hf"></a>\n\n**其他 HF 文件核查：排除项也列真实 weights/config。** 下表仅压缩展示代表文件；每行API快照包含完整、未省略的siblings列表与revision。`none`表示所查仓库未声明license，不是许可推断。\n')
skip={'brandonyang/dp-libero-checkpoints','brandonyang/dp-metaworld-checkpoints','iFaz/diffusion-libero_goal-seed3-15000','iFaz/diffusion-metaworld_medium-seed42-half','nvidia/Cosmos-Policy-LIBERO-Predict2-2B','mmurray-ms/Cosmos-Policy-MetaWorld-Predict2-2B'}
rows=[]
for p in sorted((R/'hf').glob('*/metadata.json')):
    m=read(p);mid=m['id']
    if mid in skip:continue
    fs=m.get('siblings',[])
    weights=[x for x in fs if x['rfilename'].endswith(('.safetensors','.pt','.pth','.bin')) and not any(w in x['rfilename'].lower() for w in ['optimizer','rng','normalizer','preprocessor','postprocessor'])]
    configs=[x for x in fs if x['rfilename'].endswith(('.json','.yaml','.yml')) and any(w in x['rfilename'].lower() for w in ['config','metadata','manifest'])]
    if any(x['rfilename'].endswith('manifest.ocdbt') for x in fs):
        weights=[x for x in fs if '/d/' in x['rfilename'] and (x.get('size') or 0)>1000000]
    def preview(items,n=3):return '<br>'.join(f"`{x['rfilename']}` ({x.get('size','?')} B)" for x in items[:n])+(f'<br>另{len(items)-n}个；详见快照' if len(items)>n else '') or '未列出'
    rows.append([link(mid),preview(weights),preview(configs),str(m.get('gated'))+' / '+str(m.get('cardData',{}).get('license')),f"[全部{len(fs)}项]({p.relative_to(O)})"])
out.append(table(['仓库','weights（代表）','config（代表）','gated / card license','全 listing'],rows))
out.append(table(['容易误判的项','依据'],[
    ['SmolVLA','lerobot双边config明确SmolVLM2-500M-Video-Instruct；两个model.safetensors均906,712,520 B。无检查点SR记录。'],
    ['Evo','Evo1 config的InternVL；EVO-Depth是在同VLM骨干上加入depth；最新Evo-World仓库仍InternVL3，MW权重不构成新增非VLA。'],
    ['ALAM+π0','两个card直接声明pi0 post-training。LIBERO98.05→98.1%；MW85.0是difficulty-macro，episode-weighted为86.8%；当前camera0.71另一次run为86.4%episode/85.366883%macro。不可混合统计口径。'],
    ['RLinf π0.5','LIBERO card重复论文SFT77.1→RL98.3%；不得把98.3当这个SFT权重。MW仓库没有README报告成功率。配置部分依赖RLinf代码，不是每个repo都有config.json。'],
    ['X-VLA','base_config是Florence-2。forge.safetensors仅含全部可训练参数，另需公开基础模型2toINF/X-VLA-pt固定revision c1c4a64a7e03ac5b95c468bf1578f3d03651b53b中的冻结参数。README保留原始历史评测协议（LIBERO2000/MW1000 episodes），未报SR；70k/82k是训练步数。'],
    ['CRT','[原论文](https://arxiv.org/abs/2602.01158)明确是对VLA输入图像做restoration；模型文件不是完整action策略。'],
    ['未知MW-only','Master210、hjlkhjkl等确有文件，但未建立同一非VLA家族的LIBERO对应；Rhx11111、A999baby、limiqsqsqdq、MikeDaisy为空/占位仓。不能根据名字补出架构。'],
    ['SPUR / TinyVLA','SPUR config是Qwen2.5-VL，TinyVLA本身含VLM；未因名字含diffusion而计入非VLA。'],
]))

out.append('**2024–2026 论文/代码发布复查。** 详细 pass/fail 与已摘录数值在 DECISION.md；这里保留排除证据入口。\n')
out.append(table(['方法','主要来源','发布/适用性核查'],[
    ['QueST (2024)','[论文](https://arxiv.org/abs/2407.15840) / [官方](https://github.com/pairlab/QueST)','README只有数据、train/eval、本地checkpoint；API releases=[]。不能把论文baseline DP的双基准成绩变成社区DP的成绩。'],
    ['PRISE (2024)','[论文](https://arxiv.org/abs/2402.10450) / [官方](https://github.com/FrankZheng2022/PRISE)','代码可训练双基准，未核实双边发布权重；论文观测窗口含history。LLM-style不是LLM backbone，但仍不满足R1/R3。'],
    ['VQ-BeT (2024)','[官方](https://github.com/jayLEE0301/vq_bet_official)','README quick start链接Google Drive的goal-conditioned Kitchen模型。没有理由把此文件标作LIBERO/MetaWorld；BAKU压缩目录也没有vqbet。'],
    ['STAR (2025)','[官方](https://github.com/iLearn-Lab/ICML25-STAR)','README Checkpoints/Models明确尚不包括预训练权重，release TODO未勾选；releases=[]。'],
    ['MGP (2026)','[官方](https://github.com/Coondinator/masked_generative_policy)','README仍说完整训练/推理待发布；未找到双边weights，releases=[]。long-horizon版本不断修改全局历史轨迹，非当前obs独立。'],
    ['FDP (2025/2026)','[论文](https://arxiv.org/abs/2512.21898) / [官方](https://github.com/Chaoqi-LIU/fdp)','LIBERO、MW论文实验存在；README/目录/release未见成对checkpoint下载；releases=[]。'],
    ['DiffuView (CVPR2026)','[作者项目](https://github.com/zkzwork/diffuview)','index.html的Code为占位链接；HF DiffuView search空；releases=[]。项目成功率是方法表，不是release认证。'],
    ['AutoSpeed (2026)','[官方](https://github.com/tinda24/autospeed)','公开sim download为Telon1/autospeed_alohasim_ckpt；real为cobot_magic。训练/eval_libero.py和eval_metaworld.py只是代码，releases=[]。'],
    ['LIFD (2026-09)','[论文](https://arxiv.org/abs/2609.19796)','recurrent scene memory是方法核心，直接违反当前obs条件；无需借重模型潜力强行通过。'],
    ['RoMAN-Flow (2026-08)','[官方](https://github.com/konnyaku28/RoMAN-Flow)','SmolVLM 500M，MW release列为后续；VLA排除。'],
    ['TriVLA (2025)','[论文](https://arxiv.org/abs/2507.01424)','Eagle2 VLM加视频branch；不是非VLA。'],
    ['FreqPolicy (2025)','[OpenReview](https://openreview.net/forum?id=NKNryrCGYn)','不同实验接不同policy，MW的3D/flow与LIBERO的OpenVLA不能作为相同非VLA family release。'],
    ['DataMIL (2025/2026)','[论文](https://arxiv.org/abs/2505.09603)','LIBERO用Octo，MW用环境state MLP；不是同一可视觉cache的大策略。'],
    ['DILLO (2026)','[论文](https://arxiv.org/abs/2603.23149)','是根据policy latent和planned actions做steering的LLM student，不是合格无LLM动作模型。'],
]))
out.append('GitHub Releases API已核查且空：'+', '.join(f'[{repo}](raw/releases_{repo.replace("/","__")}.json)' for repo in ['pairlab/QueST','iLearn-Lab/ICML25-STAR','Chaoqi-LIU/fdp','Coondinator/masked_generative_policy','ZibinDong/cocos','zkzwork/diffuview','tinda24/autospeed'])+'。这仅排除GitHub Release assets；同时检查README外链，不能把空release列表当作全网无权重证明。\n')
out.append('**WAM / 非 VLA 边界补查（未建立双基准交集，不计候选通过）。**\n')
out.append(table(['家族','来源 / 为什么未晋级'],[
    ['PAD / Prediction with Action','[官方](https://github.com/Robert-gyj/Prediction_with_Action)：链接yjguo/pad_bridge_pre，Bridge/MetaWorld配方；未确认LIBERO权重。'],
    ['VPP','[官方](https://github.com/roboterax/video-prediction-policy)：发布CALVIN action policy和SVD robot，未确认LIBERO+MW成对发布。'],
    ['Grounding Video Models to Actions','[官方](https://github.com/video-to-action/video-to-action-release)：发布视频模型与探索训练代码，不足以验证完整policy在两边的权重+配置。'],
    ['DynaMo / Premier-TACO','[DynaMo](https://github.com/jeffacce/dynamo_ssl)、[Premier-TACO](https://github.com/PremierTACO/premier-taco)：表征预训练或encoder checkpoint不是双边完整policy；Canonical另行审查且历史/goal不合。'],
    ['GEVRM / FastWAM / DreamZero / OpenWAM / FLOWER','名称/模型枚举和论文关键词搜索未核实双边完整发布；不要以LIBERO checkpoint+MW训练支持代替交集。'],
    ['SUREFlow / MaIL','[SUREFlow论文](https://arxiv.org/abs/2607.10504)主报LIBERO/LIBERO-PRO；[发布checkpoint issue](https://github.com/tanvirnwu/SUREFlow_IROS_2026/issues)仍开放；未建立MW交集。不能仅因Mamba自动断定有跨调用state，需具体代码，但R1已有缺口。'],
]))
out.append('**来源版本与本地审查文件。** 下表树SHA来自实际GitHub API，代码文件保存在对应目录；README/HEAD可能在未来变更，审计应以本地快照为准。\n')
rows=[]
for p in sorted((O/'code').glob('*/tree.json')):
    d=read(p);repo=p.parent.name.replace('__','/')
    rows.append([f'[{repo}](https://github.com/{repo})',d.get('sha'),f'[API tree]({p.relative_to(O)})'])
out.append(table(['官方/作者仓库','tree SHA','快照'],rows))
out.append('**本轮未执行的事。** 无GPU、无仿真成功率实验、无server/remote/tether、无完整大权重下载、无git commit、无进程终止。所有新文件位于本survey2_astra目录。CPU profiler日志有CUDA设备探测失败提示，但配置为空可见GPU且ProfilerActivity.CPU，模型tensor全部在CPU；没有GPU计算。\n')
(O/'EVIDENCE.md').write_text('\n'.join(out))
print('wrote EVIDENCE.md',len('\n'.join(out)),'chars; catalog',len(catalog),'MW screen',len(mw))
