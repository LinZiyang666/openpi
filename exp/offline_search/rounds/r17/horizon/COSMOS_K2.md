# Cosmos K2（含清理步，实际NFE1）：预注册配对分析

2026-10-10。**总体不通过本研究的“2pp非劣＋实测成本”采用门；保留K5。** 两主suite均500对完整匹配，spatial点差+0.2pp、libero_10点差−2.0pp；24项多重校正后两套件CP下界均低于−2pp。实际NFE/decision由5降至1，但可归因的服务时间成本门尚未具备数据资格。这里的“不通过”表示证据不足，**不等于已经证明损失超过2pp**。

## 数据、配对与审计

| 来源 | 本次使用与核验 |
|---|---|
| K5原段 `...100533_5f558380` | 1726个accepted结果，timan108×24；保留真实任务失败。[原段journal:1](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_100533_5f558380/driver/journal.jsonl:1) |
| K5续段 `...130639_a8ca2043` | D已生成封存的合并SUMMARY：2000/2000，PASS；本次采用其merged driver journal/per_step，而非重新拼凑或把缺局补0。[K5合并SUMMARY:38](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/SUMMARY.json:38)；[合并清单:27](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/RESUME_MERGE.json:27) |
| K5封存 | 核验46个封存文件/原段pin全部SHA256一致；续段274局原始journal与merged行集合相等，无重叠、无丢失。源段一次step0 EGL错误已由同一初态重试解决，错误attempt不作为任务失败。[封存文件:1](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/RECOVERED_RESULT_SEAL.json:1)；[错误恢复:6](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/RESUME_MERGE.json:6) |
| K2 `hs2_cosmos_h100_k2incl_c16_t107_20261010_115120` | 1000/1000，PASS，spatial/l10各483/500；零error attempts。[K2 SUMMARY:31](/data/mw_finetune/eval/hs2_cosmos_h100_k2incl_c16_t107_20261010_115120/SUMMARY.json:31)；[K2 journal:1](/data/mw_finetune/eval/hs2_cosmos_h100_k2incl_c16_t107_20261010_115120/driver/journal.jsonl:1) |
| 配对键 | `(suite, canonical task_name, original_init_idx)`；由各自EPISODE_PLAN解析task_uid末尾的task_id/init，suite用experiment。1000对原始init hash逐一相同，task_id亦相同；两suite各10任务、每任务idx0..49完整。不同臂task_uid前缀不直接相连。[K5计划:1](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/EPISODE_PLAN.json:1)；[K2计划:1](/data/mw_finetune/eval/hs2_cosmos_h100_k2incl_c16_t107_20261010_115120/EPISODE_PLAN.json:1) |
| 模型与服务器 | 两臂均H100 80GB、B1串行/replica、2 replicas、C16、policy seed195。内容manifest所有非R17代码digest相同（含权重、官方源码、配置资产）；4个R17文件digest改变：manifest/qualify/serve/staged，是K2接口与记账版本差异，完整差异保留在分析JSON。不能把聚合model_identity变化误当成换权重。[K5服务器证据:1](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/SERVER_ATTESTATIONS.json:1)；[K2服务器证据:1](/data/mw_finetune/eval/hs2_cosmos_h100_k2incl_c16_t107_20261010_115120/SERVER_ATTESTATIONS.json:1) |
| 运行偏离预注册 | 预注册K2为timan108×12，**实际为timan107×24**；K5原段timan108×24、续段timan107×8。渲染GPU按本次输入：A5000 EGL vs GTX1080 EGL。已有CPU probe证明simulator/source/init绑定一致，**没有证明跨GPU渲染像素一致**。本次不修改STUDY追认该偏离。[K2部署:50](/data/mw_finetune/eval/hs2_cosmos_h100_k2incl_c16_t107_20261010_115120/campaign.json:50)；[K5续段部署:514](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/campaign.json:514)；[D的CPU probe:731](/home/weiland/projects/openpi/exp/r17_integration/handbacks/HANDBACK_COSMOS.md:731) |

原结果及STUDY只读；无GPU、远程调用、campaign操作或凭据访问。新产物为本文、[CPU分析脚本:1](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/analyze_cosmos_k2.py:1)、[全部统计/1000对身份/输入SHA256:1](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/COSMOS_K2_ANALYSIS.json:1)。脚本将decision、execution_ack、client_timing区分，避免把一次调用及其ack双计；accepted的成功和正常失败都保留，执行控制步以ack求和并核对attempt.n_steps（不含10步settle）。

## 冻结方法与名义95%敏感性

采用 [STUDY判据:125](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/STUDY.md:125)：`g=K2成功且K5失败`、`l=K2失败且K5成功`，`Δ=(g−l)/n`。**主结论保持24项family-wise 95%下界，不因这里只报告Cosmos就把族缩成2项。** CP差下界为 `BetaPPF(a;g,n−g+1)−BetaPPF(1−a;l+1,n−l)`，`a=.05/(2×24)=.0010416667`（0/n边界用0/1）。这是预注册的保守配对CP构造，沿用其二项工作模型，不冒称另一个“精确配对差检验”。

Bootstrap在每任务内成对有放回抽样，10任务样本数各50不变；10,000次、PCG64 seed20261010、canonical task/init排序、两臂共用抽样索引，取`.05/24=.0020833333`分位，线性分位插值。主CP和bootstrap下界都须**严格大于−2pp**，且两个suite分别满足。每个cohort重置同一seed；274/226等子组保留各任务原有样本数，仅作描述，无新增确认性检验。主表见文末。

| suite | gains / losses | 双方失败 | McNemar精确双侧p（描述） | 未校正单侧95% CP LB，pp | 未校正单侧95% bootstrap LB，pp |
|---|---:|---:|---:|---:|---:|
| libero_spatial | 5 / 4 | 13 | 1.000000 | -1.71 | -0.80 |
| libero_10 | 4 / 14 | 3 | 0.030884 | -4.43 | -3.40 |

未校正下spatial可跨过2pp门，但这不替代冻结的24项规则。libero_10未校正McNemar p=.030884仅表示描述性不对称；未作多重校正，也不能证明损失超过2pp。δ=0敏感性两suite均不过；δ=5pp时spatial可过而l10 CP下界仍低于−5pp，总体仍不过。不用跨suite合并的−0.9pp平均差掩盖l10。

## 渲染节点混杂：同timan107的274局 vs 其余

| 子集 | n pairs | K5成功/SR | K2成功/SR | gains/losses | Δ，pp | 未校正95% CP / bootstrap LB，pp |
|---|---:|---:|---:|---:|---:|---:|
| K5续段、双方timan107（全部为l10） | 274 | 268/274（97.81%） | 262/274（95.62%） | 3/9 | -2.19 | -5.92 / -4.38 |
| l10其余：K5 timan108 → K2 timan107 | 226 | 225/226（99.56%） | 221/226（97.79%） | 1/5 | -1.77 | -5.08 / -3.54 |
| spatial全部：K5 timan108 → K2 timan107 | 500 | 482/500（96.40%） | 483/500（96.60%） | 5/4 | +0.20 | -1.71 / -0.80 |
| 两个主suite其余合计（仅描述） | 726 | 707/726（97.38%） | 704/726（96.97%） | 6/9 | -0.41 | -2.04 / -1.24 |

274局＝task34的24个init（2、9、28..49）＋task35..39各50。双方timan107的子集仍净少6次成功；l10跨节点226局净少4次，合计净少10。因此**不能将l10下降全部解释为A5000→GTX1080切换**，也不能据此证明减步是唯一原因：274局集中在后六项任务、不是随机节点对照，续段时段/8-worker负载又与K2的24-worker不同。spatial没有双方timan107的子集可验证。其余726局混合两个suite，不能与274局直接比较来估计“节点效应”。

## 成本：NFE已确认下降，纯服务时间门未定

所有主集K5的11,803次decision均`actual_nfe=5 / stage_calls[2]=5`；K2的11,921次均为1。名义“含清理步”严格按上游num_steps=2：clean预算扣除后落入sigma80单次分支，**没有额外sigma4 clean评估**，不能报NFE2。依据 [K2语义:371](/home/weiland/projects/openpi/exp/r17_integration/handbacks/HANDBACK_COSMOS.md:371)、[封存sampler:61](/data/mw_finetune/eval/hs2_cosmos_h100_k2incl_c16_t107_20261010_115120/SUMMARY.json:61)。

| suite/子集 | 决策次数 M：K5→K2 | 决策/局：K5→K2 | 控制步/局：K5→K2 | NFE/控制步：K5→K2 | NFE/控制步减少 |
|---|---:|---:|---:|---:|---:|
| spatial | 3550→3543 | 7.100→7.086 | 106.342→106.450 | 0.33383→0.06657 | 80.06% |
| libero_10 | 8253→8378 | 16.506→16.756 | 256.478→260.430 | 0.32178→0.06434 | 80.01% |
| 同节点274局 | 4676→4743 | 17.066→17.310 | 265.153→269.186 | 0.32181→0.06431 | 80.02% |
| 主集1000局（描述合计） | 11803→11921 | 11.803→11.921 | 181.410→183.440 | 0.32531→0.06499 | 80.02% |

每decision NFE下降80%；全主集NFE总计59,015→11,921（−79.80%），决策数反而+1.00%。l10控制步/局增加，不能用“更早失败少走路”解释这里NFE/控制步的下降。

| suite/子集 | 实测server host ms/call：K5→K2 | 观测降幅 | host ms/控制步：K5→K2 | mean episode seconds：K5→K2 |
|---|---:|---:|---:|---:|
| spatial | 5814.2→585.6 | 89.9% | 388.19→38.98 | 46.23→23.56 |
| libero_10 | 3533.5→1034.3 | 70.7% | 227.40→66.55 | 70.59→44.16 |
| 同节点274局 | 1430.2→1043.3 | 27.1% | 92.05→67.09 | 40.58→43.58 |
| l10跨节点226局 | 6282.9→1022.6 | 83.7% | 404.30→65.84 | 106.97→44.87 |

这些是所有实际policy decision的`server_timing.infer_ms`算术均值，**不是无竞争GPU kernel时间或已剥离队列的service_ms**。两臂主集全部23,724条decision都`pricing_eligible=false`，`service_ms`、`gpu_service_ms`、`queue_wait_ms`全为null；无法验证等负载、剥离排队/共卡干扰，故按预注册**成本门：NFE部分通过，服务成本部分未定**。尤其同节点274局host/call只下降27.1%，episode均时反而40.58→43.58秒，不能把总体大幅host下降宣称为纯减步端到端加速。数据字段见 [K5逐步成本:1](/data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043/driver/per_step.jsonl:1)、[K2逐步成本:1](/data/mw_finetune/eval/hs2_cosmos_h100_k2incl_c16_t107_20261010_115120/driver/per_step.jsonl:1)；语义见 [pricing字段:369](/home/weiland/projects/openpi/exp/r17_integration/common/serving.py:369)。

K2整run driver 1621.54秒（0.450h），K5合并driver 7393.37秒（2.054h）覆盖四suite及恢复段，既不等样本量也不等worker/负载；仅作运维记录，不能拿二者比值当节省比例。

## 逐任务审计与l10下降来源

每行50对；g/l与init详见分析JSON。逐任务SR仅描述，不作逐任务非劣声明。

| suite/task_id | canonical task | K5成功 | K2成功 | Δ，pp | g/l | 双方timan107的n |
|---|---|---:|---:|---:|---:|---:|
| libero_spatial/0 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 50/50 | 50/50 | +0.00 | 0/0 | 0 |
| libero_spatial/1 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 50/50 | 50/50 | +0.00 | 0/0 | 0 |
| libero_spatial/2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 50/50 | 49/50 | -2.00 | 0/1 | 0 |
| libero_spatial/3 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 47/50 | 48/50 | +2.00 | 2/1 | 0 |
| libero_spatial/4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 48/50 | 49/50 | +2.00 | 1/0 | 0 |
| libero_spatial/5 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 41/50 | 42/50 | +2.00 | 2/1 | 0 |
| libero_spatial/6 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 50/50 | 49/50 | -2.00 | 0/1 | 0 |
| libero_spatial/7 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 48/50 | 48/50 | +0.00 | 0/0 | 0 |
| libero_spatial/8 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 48/50 | 48/50 | +0.00 | 0/0 | 0 |
| libero_spatial/9 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 50/50 | 50/50 | +0.00 | 0/0 | 0 |
| libero_10/30 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49/50 | 49/50 | +0.00 | 1/1 | 0 |
| libero_10/31 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 50/50 | 49/50 | -2.00 | 0/1 | 0 |
| libero_10/32 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 50/50 | 50/50 | +0.00 | 0/0 | 0 |
| libero_10/33 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 50/50 | 47/50 | -6.00 | 0/3 | 0 |
| libero_10/34 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 47/50 | 48/50 | +2.00 | 2/1 | 24 |
| libero_10/35 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 50/50 | 50/50 | +0.00 | 0/0 | 50 |
| libero_10/36 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 49/50 | 46/50 | -6.00 | 0/3 | 50 |
| libero_10/37 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 50/50 | 50/50 | +0.00 | 0/0 | 50 |
| libero_10/38 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 48/50 | 46/50 | -4.00 | 1/3 | 50 |
| libero_10/39 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 50/50 | 48/50 | -4.00 | 0/2 | 50 |

最大净损失是task33“黑碗放底抽屉并关闭”和task36“白杯放盘、布丁放右侧”，各少3成功（各−6pp）；task38“双摩卡壶上炉”和task39“黄白杯放微波炉并关闭”各少2（各−4pp）。task31少1，task34多1抵消；五个下降任务净少11，提升任务补1，得到l10净少10。task33的3次loss在跨节点部分；task36/38/39均属双方timan107，所以同节点仍有主要损失。

## 与早期Cosmos NFE实验的关系

9月旧研究spatial K5/K2/K1为98.6/97.4/97.4%，各500；l10 K5/K2为98.0/97.2%，K1未跑。[旧终表:49](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:49)
旧日志把基类dummy的“k+1 NFE”用于Policy，并且l10控制在4090、K2在H100；本次逐调用确认K2实际NFE1且服务器同H100，但仍有渲染节点/负载混杂。[旧NFE说明:46](/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md:46)
因此旧“几乎无损”只作历史点估计背景；不与本次池化，也不能替代当前配对的24项校正非劣判决。

## 最终判决表

下界均为**24项family-wise 95%单侧下界**（每项α=.05/24），单位pp；上文另列未校正单侧95%下界。

| suite | n pairs | K5 SR | K2 SR | Δ，pp | CP LB，pp | bootstrap LB，pp | 2pp非劣判决 |
|---|---:|---:|---:|---:|---:|---:|---|
| libero_spatial | 500 | 96.40% | 96.60% | +0.20 | -2.77 | -1.60 | 未通过：证据不足 |
| libero_10 | 500 | 98.60% | 96.60% | -2.00 | -5.78 | -4.40 | 未通过：证据不足 |

**总体判决：未通过2pp非劣采用门，保留K5；不是证明K2必然损失超过2pp。** spatial只有bootstrap过门，CP不过；l10两者均不过。

**成本减少：** NFE/decision 5→1（−80%），spatial/l10 NFE/控制步分别−80.06%/−80.01%；观测host/call下降89.9%/70.7%，但服务时间未剥离队列与负载，预注册服务成本门未定。

**混杂：** 同timan107的274对268→262（−2.19pp），l10其余226对225→221（−1.77pp）；同节点仍掉点，不能全部归因于换渲染GPU，但子集任务/时段/worker数不平衡，不能作纯K因果结论。

**主要下降任务：** task33底抽屉黑碗、task36白杯＋布丁各−3成功；task38双摩卡壶、task39微波炉杯各−2；task31−1与task34+1抵消。
