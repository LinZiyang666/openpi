<task>
Good work on the size grid. Change one thing before arming: the size lanes must not put all workers on timan107. Q has added
node-split fleets and timan1 (see exp/r17_integration/REQUESTS/Q_to_P6b.md, HANDBACK_Q.md, and the running Fast-WAM continuation
fastwamc2350933912* which uses timan107=16, local=16, timan1=16 with 4 GiB per worker; timan1 GPU 3 only, 6 GiB per worker,
max 16). The two lanes run concurrently, so their fleets must fit together:
- 4090 lane (DP sizes 50..400): 32 workers = timan107=16, local=16
- H100 lane (Fast-WAM sizes then Cosmos sizes): Fast-WAM 48 = timan107=16, local=16, timan1=16; Cosmos: keep its worker count
  with a split that fits (prefer timan1/local).
Observations are byte-identical across nodes (Q local smoke, T1 672-array check); record node per worker. Also note: I (the
coordinator) mistakenly SIGTERMed your CPU prepare process for cszeh at ~20:5x — make sure cszeh/cszedp preparations are complete
and consistent (re-prepare if needed). Then re-prepare both size lanes with these fleets (CPU), validate, and ARM them yourself
(nohup the two run scripts so they wait for csl10dp / csl10h to finish and PASS, then dispatch). Disk guard: timan107 now has ~32 GB
free; P6's cleanup removes remote copies after verified local seal. Give updated duration estimates.
</task>
<compact_output_contract>
End with: fleets per lane, re-prepare/validation result, the two armed PIDs and their wait conditions, updated durations.
</compact_output_contract>
<action_safety>
CPU only until your armed scripts dispatch after predecessors PASS. Do not touch running lanes (DP continuation on 4090,
Fast-WAM continuation on H100). No pkill -f bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials.
timan1 GPU 3 only.
</action_safety>
