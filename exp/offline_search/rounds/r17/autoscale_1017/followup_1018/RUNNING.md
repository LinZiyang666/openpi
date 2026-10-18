当前状态（2026-10-18 04:35 CDT）：候选全部写好、尚未启用；共享9个runtime源文件仍匹配BEFORE_SOURCES.json。fairness.py是新增但尚未进入live bundle的文件。ACTIVATION.json标记STAGED_NOT_ACTIVE。

当前测试：ops_before.xml=495通过/36既有失败/0超时，整套2275秒；FAIR_11.xml=18通过。第一轮ops_after只作诊断，因为启动后补了跨lane memory_sample_valid与critical优先级。run_regression.py在这一轮后还会串行跑其余7套before/candidate；run_final_ops.py等REGRESSION_COMMANDS.json有15条后自动开始最终完整ops，输出ops_after_final.xml及OPS_FINAL_COMMAND.json。

所有最终候选源代码已冻结，SHA在ACTIVATION.json。不要再改候选生产代码，除非测试发现真实问题。不要以第一轮ops_after代替最终ops_after_final。其它7套开始时会使用最终代码。全部CPU，有cpu_only远端/GPU命令保护与逐例超时；当前没有启动正式任务或写正式campaign目录。

完成顺序：

1. 等OPS_FINAL_COMMAND.json和所有套件完成；执行 `PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r17/autoscale_1017/followup_1018/report.py`，检查逐例差异与源码依赖哈希，必须PASS。
2. 执行同目录activate.py；它先验证所有原始/候选SHA和文档基线，再发布9个runtime文件及文档，最后标记ACTIVE。新文件fairness.py已存在，bundle首先切换。不会写campaign或chain。
3. 直接导入实际共享树执行整个autoscale_1017/tests（预期101项，其中7项指定prepare在test_new_chains.py内；不能再使用candidate覆盖层）。使用cpu_only插件、CUDA_VISIBLE_DEVICES空、JAX_PLATFORMS=cpu、BLAS单线程、AUTOSCALE_TEST_TIMEOUT=180。独立basetemp=followup_1018/postapply_tmp，JUnit=followup_1018/POST_APPLY.xml，日志同目录postapply.log。必须先等其它大计划测试完成，不并行大读。
4. 运行longseg_1017的单元测试（此前27项），同样CPU guard，独立basetemp与JUnit=followup_1018/LONGSEG_UNIT.xml。不需要远端或重新launch；本轮未改sol的科学文件。
5. 执行同目录update_verification.py更新根VERIFICATION.json与最终文档。该脚本验证POST_APPLY全绿、longseg单元全绿、源码SHA匹配ACTIVE、7项dryrun全PASS且remote=launch=0。
6. 检查最终表、7项maxima、ACTIVE时间、文档及源码一致，给用户简短中文handback。deadline约05:45；预估最终ops于05:40左右完成。若超时，主动告知树仍是上一版完整实现（直到activate成功），不要宣称已切换。

首批7项期望：oc1固定44/34；fw_r_p_m、gr_r_p_m、dp_r_u_m为332/440；dp_r_b_l为83/104；pi_r_f_l、gr_r_u_l的R16为timan107=104。当前61个已有R17定义都未PREPARED且没有worker_hard_limits；fixed continuation是唯一小数量例外。metadata/任务计划只读，干运行输出重定向到交付目录。

事故取证已完成：本轮仅两次获准的timan107只读ls/stat/sha，无其他远端调用。唯一远端遗留/tmp/openpi_autoscale_e9f9bbe5134a2fa64b1366ba3c904f8860dadb57fc777a1177a56241a1f97e7d.tar，409600 bytes，ctime 2026-10-18 00:31:42.569994496 -0500；sha94d4c3086976beb0c5214a31bcee44c9872ca46b74625cd23bec83630690e2a9。未解包，未联系H100、未写科学岛或冻结payload。无法证明同名/tmp文件此前不存在。原测试曾短暂开本地rsync daemon23118，helper随后清理自身daemon/临时导出，必须如实披露。此次不删除任何事故遗留。incident/INCIDENT.md和INCIDENT.json有全部细节；LOCAL_FILES.json列48项本地遗留。


URGENT update 2026-10-18 ~05:40 CDT:
- User interrupted because fa_ls_fw_r_p_m failed on recycled renderer.pid under identical worker_id. Old background exec jobs were terminated by that interruption; ops_after_final stopped at25%, no final XML. Do not use it or the old finish.py runner.
- PUBLICATION_HOLD exists. Shared10runtimefiles still previousaccepted; generation candidate only staged.
- Fix source: generation_candidate/worker.py maps core src/openpi/conductor/worker.py through generation_hook/sitecustomize.py. Only5lines added afterObservercreation: UUID communicationworker_id; Observer slotidunchanged. Driver/probevalidatorunchanged.
- Red reproduction GENERATION_BEFORE.xml; passing GENERATION_AFTER.xml; 3real R17/R15/R16 generation/pin tests GENerATION_PINS.xml; generation+8reset tests GENERATION_FINAL.xml11passed.
- New reset_zero.py (unused until owner runs) backs up entire closedzeroaccepted directory and re-prepares sameplan. Exactcommand URGENT_FIX.md. No actualreset/prepare/launch executed.
- ACTIVE runner: run_combined_regression.py session32637; 8after suites with fairness9source+coregeneration overlay, against existing *_before.xml baseline. Log combined_runner.log and combined_regression/*_after.log. Began ~05:30, ETA full8~06:10.
- ACTIVE waiter: finish_combined.py session17674. Waits8reports, report_combined.py; full112candidateCPUtests; resolvesHOLDb yrename onlyaftertestsPASS; activate.py publishes10runtimefiles anddocs; full112direct-tree tests; longseg27; update_verification.py.
- Existing9candidatecode/fairnessremainfrozen unchanged; ACTiVATION.json nowalso pinsgenerationcorebefore/after. reset_zero/tests/docs are new and notinworkerbundle.
- ReadonlylocalR15fa_pi_f_u_o evidence:6of8subjobs have LEGACY_AUTOSCALE.json max104 andoldcoreSHA a7b29e...; last2stillpreparing. NoPIDinlegacyLIBEROprobe, stockR15driverdoesnotrejectthissameerror; R16probe_conflictstillstrict. Ownerneeds repeat formal_a_queue dispatch prepare --name fa_pi_f_u_o afterpublish fornewfairnesspayload, ifnotlaunched. Do not edit these dirs.
