"""Generate reviewable operating plan from the concrete, runnable catalog."""
import json
from pathlib import Path


def main():
    q=Path(__file__).resolve().parent
    jobs={j['slot']:j for j in json.loads((q/'SLOTS.json').read_text())['slots']}
    dp_proof=q/'receipts/DP_NATIVE_CPU.json'
    dp_ready=dp_proof.is_file() and json.loads(dp_proof.read_text()).get('pinned_slot_gate') is True
    cpu_index=q/'receipts/CPU_SLOTS.json'
    cpu_results={r['slot']:r for r in json.loads(cpu_index.read_text()).get('results',[])} if cpu_index.is_file() else {}
    orders={
        'local':['slot_G1_pi05_mw_A','slot_G1_pi05_mw_B',
            'slot_G2_groot_mw_A','slot_G2_groot_mw_B','slot_G2_groot_mw_A_k2','slot_G2_groot_mw_A_k1',
            'slot_G2_groot_mw_C_B2','slot_G2_groot_mw_R_replicas',
            'slot_G3_pi05_mw','slot_G3_groot_mw','slot_G1_pi05_mw_C_tail',
            'slot_G2_dp_libero_A','slot_G2_dp_libero_B','slot_G2_dp_libero_R_replicas','slot_G3_dp_libero',
            'slot_G2_groot_mw_final60000_A','slot_G2_groot_mw_final60000_B'],
        'h100':['slot_G2_fastwam_mw_A','slot_G2_fastwam_mw_B',
            'slot_G2_fastwam_libero_A','slot_G2_fastwam_libero_B','slot_G3_fastwam_mw',
            'slot_G3_fastwam_libero','slot_G2_fastwam_mw_R_replicas',
            'slot_G2_dp_mw_final100000_A','slot_G2_dp_mw_final100000_B',
            'slot_G2_dp_mw_A','slot_G2_dp_mw_B','slot_G2_dp_mw_R_replicas','slot_G3_dp_mw',
            'slot_G2_fastwam_mw_final16001_A','slot_G2_fastwam_mw_final16001_B',
            'slot_G2_fastwam_mw_C_B2','slot_G2_fastwam_libero_C_B2']}
    assert set(sum(orders.values(),[]))==set(jobs)
    text='''# R17 Q qualification plan

GPU fix4 regeneration after Q's allowed-root archive transfer fix, 2026-10-09.
**This refresh is CPU-only.** The coordinator's GR_k2, π_k2 and DP-LIBERO
evals are running on the 4090 lane. Preserve their lock, ports and workers;
the ready queue requires GPU admission before any model load. H100 waits for both DP-LIBERO
training (estimated20:30 CDT) and Fast-WAM training (estimated22:15 CDT) to exit.
The earlier DP completion does not free the lane while Fast-WAM remains active.
Run each GPU slot manually in the coordinator's tmux; clocks do not grant admission.
The ledger identifies this local 4090 as **48 GB** and H100 as **80 GB**.

Every wrapper is one command from the coordinator host, uses a fresh run under
`/data/mw_finetune/qualification/runs`, and accepts `--preflight` (strictly CPU)
or `--share-eval` (only the catalog's eligible memory budgets). All H100
dispatch, worker launching, checksum transfer and receipts are automatic.
The 34 wrappers include two retired Fast-WAM C/B2 commands that always return
CPU-only PARTIAL because B>1 is disabled; 32 wrappers remain GPU candidates.

```bash
bash exp/r17_integration/qualification/slot_G1_pi05_mw_B.sh
bash exp/r17_integration/qualification/slot_G2_groot_mw_B.sh
```

For a memory-eligible concurrent slot while the 500-episode evaluator holds the
GPU lock, append `--share-eval`. The default takes `/tmp/r17_gpu.lock`
nonblocking. Every slot also takes its Q lock, rejects live GR00T/Fast-WAM/DP
training via CPU process census, checks only its own assigned ports, then reads
free GPU memory before loading a model. A busy lock/port, training, insufficient
memory or the documented local driver/library mismatch returns PARTIAL.
Q performs no system repair or changes to tmux/training processes. Every live
slot waits up to60 seconds for its Q policy/driver/backend ports to release;
H100 dispatch also waits for the local Q relay port23158. The wait is inside
the original hard deadline, records initially busy ports/elapsed time/status,
uses listener-compatible SO_REUSEADDR, and never signals the port holder.
An occupied port still returns PARTIAL if the bounded wait expires.

The table's peak numbers are **conservative planning estimates, not GPU
measurements**. Sharing budgets reserve 20 GiB for formal GR00T on 4090, 34 GiB
for formal Fast-WAM on H100, and 6 GiB combined margin. Admission additionally
requires peak+4 GiB free. Fast-WAM LIBERO remains exclusive because its two-view
activation budget lacks a measured certificate. If the formal evaluator exceeds
its planning budget, free-memory admission defers the Q slot. Shared runs cannot
provide isolated latency/pricing certificates.

'''
    for host,order in orders.items():
        text+=f"## {'4090 / weilandserver → timan107' if host=='local' else 'H100 → timan108'}\n\n"
        text+='Run A before B for each checkpoint; require A+B PASS before replicas/G3. Conditional rows wait for their named dependency, so they do not block earlier ready work. Fast C/B2 is retired and CPU-only. G3 and the tail retain explicit PARTIAL scopes.\n\n'
        text+='| Order | Single command script | Expected minutes | Estimated peak GiB | Minimum free GiB | Can share formal eval by budget | Prerequisite |\n|---:|---|---:|---:|---:|---|---|\n'
        for i,name in enumerate(order,1):
            job=jobs[name]
            share='CPU only; retired B>1' if job['phase']=='batch_disabled' else ('Yes, explicit `--share-eval`' if job['share_eval_safe'] else 'Exclusive')
            prerequisite=('Retired; CPU only' if job['phase']=='batch_disabled' else
                ('P4 probe API' if job['phase']=='native' else
                    'P4 probe API; matching A PASS' if job['phase']=='mechanical' else 'P4 probe API; A+B PASS')
                    if job['model']=='dp' and not dp_ready else
                ('Immutable publication' if job['phase']=='native' else 'Immutable publication; matching A PASS')
                    if job.get('future_checkpoint') and (cpu_results.get(name,{}).get('reason') or '').startswith('future immutable checkpoint not yet published:') else
                'Matching A PASS' if job['phase']=='mechanical' else
                'A+B PASS' if job['phase'] in ('replica','g3','batch','tail') else 'GPU admission')
            text+=f"| {i} | [{name}.sh]({name}.sh) | {job['expected_minutes']} | {job['estimated_peak_mib']//1024} | {job['required_free_mib']//1024} | {share} | {prerequisite} |\n"
        text+='\n'
    text+='''## Pinned assets and execution islands

| Model / domain | Immutable checkpoint now | Later explicit variant |
|---|---|---|
| π0.5 MW | `/data/ckpt/pi05_metaworld_rlinf` | same fixed RLinf checkpoint |
| GR00T MW | `/data/mw_finetune/groot/4090_mt50/checkpoint-30000` | `checkpoint-60000` in the same run |
| DP MW | `/data/mw_finetune/dp/mt50_policy_100000` | same formal serving export in `final100000` slots |
| DP LIBERO | `/data/mw_finetune/dp/checkpoints/libero` (brandonyang 30k) | mechanics checkpoint only |
| Fast-WAM MW | `/data/mw_finetune/fastwam/mt50_frozen_b128/policy_sanity_004000` | `policy_016001` in the same run |
| Fast-WAM LIBERO | `/data/mw_finetune/models/fastwam_libero` | published mechanics checkpoint |

No `latest`, live trainer directory or mutable adapter file is selected.
DP MW uses the immutable formal serving export of scratch checkpoint-100000,
with diagnostic=False and pinned weight/norm/model identities. Current and
final100000 variants share that CPU seal. No trainer-root copy is made. GR60000 is now published and its native GPU A
passed. Fast016001 remains unpublished and returns PARTIAL.
Their A wrapper with `--preflight` can seal the newly published immutable assets
without any GPU use. If invoked live with published but unsealed assets, sealing
has a 180-second phase bound inside the same overall slot deadline; later B
reuses that completed seal. Partial seals are never overwritten automatically.

| Work | Python executable |
|---|---|
| Local control / π | `/home/weiland/projects/openpi/.venv/bin/python` |
| GR00T | `/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python` |
| DP local | `/data/mw_finetune/dp/env/bin/python` (3.11) |
| DP H100 | `/data/mw_finetune/dp/env/bin/python` (3.12, existing shared island) |
| Fast-WAM H100 | `/data/mw_finetune/envs/fastwam/bin/python` |
| H100 control | `/home/exouser/openpi/.venv/bin/python` (3.11) |
| Both timan workers | `/scratch/zixuans8/mw_r17/envs/sim/bin/python` (3.11.15) |
| LIBERO CPU renderer, local | `/home/weiland/projects/openpi_ext/envs/libero_sim/bin/python` (3.8.20) |

The H100 Fast/DP child PYTHONPATH prepends the Q-only h5py overlay
`/data/mw_finetune/qualification/python_deps312`; the training environment was
not altered. GR uses the local vendor tree and local Eagle tokenizer assets
sealed by exact content. The H100 runtime copy is under
`/data/mw_finetune/qualification/runtime_q1`, independently of P-owned source.
Fast-WAM stays on H100 for every slot, so no large delta/base is copied locally.
Its PYTHONPATH explicitly includes `/data/mw_finetune/vendor/FastWAM/src` and
`/data/mw_finetune/vendor/lerobot/src`, so native dependency imports use the
same sealed source trees as the loader before model construction.

## Ports, workers, artifacts and bounds

| Purpose | 4090 lane | H100 lane |
|---|---|---|
| Q public policy | `0.0.0.0:23156` → `ziyanglin.com:23156` | `0.0.0.0:23256` → `149.165.153.233:23256` |
| Q MW driver | `0.0.0.0:23157` → `ziyanglin.com:23157` | `0.0.0.0:23257` → `149.165.153.233:23257` |
| Optional replica backends | `127.0.0.1:23160/23161` (server currently binds 0.0.0.0) | `127.0.0.1:23260/23261` (same binding) |
| Q CPU relay | `0.0.0.0:23158` → `ziyanglin.com:23158` | H100 connects here; no H100 tether credentials installed |

CPU echo proofs passed for both policy/driver pairs from their actual timan
node. H100 → relay → deployed timan108 import also passed. Q reuses these lanes
sequentially; it never uses I018's 23180, formal 23150/23151/23152/23154/23159,
or the running H100 DP eval ports 23254/23255/23258/23259.
The r17_srv_dp1..4 servers and timan108 r17_dp1..4 drivers are untouched.
Ports are checked again on every live launch.
Preflight verifies exact deployed Python/spec bytes against
`runtime_code_snapshot.json`. An owner edit after handoff requires a CPU
payload/spec refresh plus new immutable asset seals where identities changed;
source drift returns PARTIAL before any GPU command. This generation was
regenerated after confirmed A3 completion and consumes the final A3/B3 code.
[SLOTS.json](SLOTS.json) pins both final review report hashes. No A3 completion
dependency remains. Any subsequent owner edit still requires deployment/spec
refresh and resealing where model identities change.
MW launches exactly two owned workers on EGL devices 0 and 1, using the
unchanged authoritative reach-v3 seed7/camera480/settle15/replan5/control160
kernel and actual task-data hashes. LIBERO uses the unchanged spatial task0,
original indices 0/1, seed7/settle10/replan5/control220 kernel, both real cameras,
and a Q context/ack shim in the existing Python3.8 OSMesa island. It is CPU
rendered even when the policy runs on H100. There is no real-environment SR
claim from the synthetic donors.

Exact installed SHA paths for both timan payloads are in
[worker_deployment.json](worker_deployment.json) and each slot specification.
Deployment uses P6 `ops/deploy.py`'s actual installer, SHA validation and atomic
new directory install. Q's bootstrap only makes that same P6 installer callable
before a package exists. Worker debug/client imports are model/simulator-free.

Current model assets are in `/data/mw_finetune/qualification/assets/`:
`pi05_mw_fix2_b3a3_v1`, `groot_mw_fix2_b3a3_v1`, `groot_mw_k2_fix2_b3a3_v1`,
`groot_mw_k1_fix2_b3a3_v1`, `dp_libero_gpu_fix3_v1`, and, on H100,
`dp_mw_gpu_fix3_v3`, `fastwam_libero_fix2_b3a3_v1`, `fastwam_mw_fix2_b3a3_v1`. Each includes sealed model/load
manifests, validated bound YAMLs, `synthetic_teacher_store`,
`synthetic_library.pkl`, `synthetic_fit`, and explicit `SYNTHETIC.json` labels.
All donors are fabricated CPU arrays (3 complete episodes / 9 decisions), with
real production geometry/identity and no teacher or quality certificate.
MW observation fixtures are separately under `qualification/observations/*_v2`;
LIBERO uses original recorded paired256 views from dataset episode1272,
frames0/1/2, with matching real states/prompts. PNG bytes, decoded RGB and NPZ
files are SHA sealed in OBSERVATIONS.json; no camera is flipped or fabricated.
These recorded inputs are independent of the explicitly synthetic teacher stores.
Native-range MW states remain distinct after GR
normalization. The original sealed synthetic stores were not rewritten.
LIBERO uses the sealed official four-suite P2 task map with canonical suite/task
names and global IDs; the exact original language remains the model prompt.
Every recipe's ordered_tasks equals the complete canonical benchmark table
(MW50 or LIBERO40), with task_table_identity. π G1 uses periodic_gap=2 and
target_ir=None; Q does not invent an IR pricing profile. DP live bundles use
K100/K2; K1 is excluded from recipes/descent and G3 augmentation.
DP K100 is native stochastic DDPM; reduced K2 is deterministic trailing DDIM,
so step reduction also changes sampler family. Native K1 study evidence is
labelled separately and cannot qualify a live retrieval/descent arm.

Run receipts, logs and model captures stay in the fresh local/H100 run path.
Closed MW physical archives are verified and installed under that run; H100
pulls are relayed through the coordinator host. LIBERO archives also contain a
P6-compatible file SHA manifest and are verified by P6's actual installer.
Local H100 receipts include its exact remote run path and recovered receipt.
Large H100 artifacts remain there for bounded follow-up CPU transfer.
MW tar sources on both timan nodes are now staged in unique
`/tmp/r17_Q_qualification/<run>/workers_<sha>_<uuid>.tar` files, SHA-checked
before publication. H100's remote LIBERO worker uses the same allowed root
with libero_<sha>_<uuid>.tar. The coordinator verifies its pull and P6 closed
manifest before deleting that exact owned source stage. Failed or corrupt
transfers retain the stage; original run archives are preserved.
The H100 relay verifies the coordinator's pull, pushes into an allowed /tmp
stage on H100, then performs a bounded CPU copy with SHA checks into the exact
Q run. It removes that incoming stage after the durable copy is verified.
No tether pull from /scratch or /data, or push to H100 /data, remains in the
qualification transfer path. Deployment pushes already target allowed /tmp.

Shell watchdog: TERM at 885 s, KILL at 890 s; stdlib fallback writes PARTIAL.
Supervisor work bound: 840 s, with bounded owned cleanup and receipt writing.
MW driver 420 s, each remote worker supervisor 500 s independently even if
tether disconnects, LIBERO worker 500 s, load readiness 180 s per instance.
PIDs and process birth ticks are recorded. Cleanup uses owned Popen handles or
receipt+birth+exact argv for remote supervisors. Direct children have a Linux
parent-death TERM signal. No process-name kill, foreign tmux, or `pkill`.
Forced stop, missing archive, rejected attempt, startup failure or timeout
cannot become PASS. The Q relay closes its owned tether children too.

H100 Q data has a 4 GB operating cap plus cleanup/archive headroom within the
5 GB addition allowance. Admission also requires the shared tree remain below
107 GB, preserving 13 GB of the 120 GB budget for a Fast-WAM save. No weights,
optimizer snapshots or large checkpoint trees are copied into Q space.

## Exact receipt interpretation

`PASS` certifies only the listed slot scope; `PARTIAL` retains evidence and
pending causes; `FAIL` records a violated assertion/model error. Exit codes are
0/2/1 respectively. [SLOTS.json](SLOTS.json) and each `specs/slot_*.json` include
the exact pass criteria and all concrete paths, ports, islands and memory
budgets. The Q integration tests exercise actual shared WS and capture/fit code.

P4's --load-spec/--observations API is now implemented. Fresh DP seals and real
CPU probes pass13/skipped0 with exact pinned ModelSpec/weight identities;
DP-LIBERO uses recorded256 pairs and DP-MW preserves diagnostic=False/formal100k.
[Q_to_P4](../REQUESTS/Q_to_P4.md) is resolved at this seam. DP-LIBERO A and B
now passed on GPU for the B1 release; preserve those receipts. Its replica slot
is ready subject to admission. Its G3 scoped PARTIAL has already been observed.

P2 resolved [Q_to_P2](../REQUESTS/Q_to_P2.md), GPU fix1, in its GPU fix2 delivery.
Both nodes have metaworld3.0.0/mujoco3.14.0/gymnasium1.3.0. Stock workers send a
full probe once per TCP connection; P2 now reuses the accepted probe only on
the same attested worker/peer/endpoint, preserving strict admission checks.
Q's replay must pass first, second and terminal pulls plus all strict rejection
cases before recording the current release ready. GR B/current+60000 now retry
with the refreshed deployment, preserving their existing native A GPU PASS.
Q preserves raw rejected pulls, accepted previous probe, peer and reason in
rejected_worker_probes.json plus driver/top-level receipts and stops promptly.
Q's GPU fix3 adds a bounded35s shutdown drain around the unchanged stock TCP
handlers, so scheduler completion cannot close existing worker sockets before
their final MSG_SHUTDOWN pulls. The remote supervisor records normal_exit versus
deadline/interrupt/nonzero status, closes only its own children, and bounds P6
snapshot production with an owned recorded PID. B requires two normal exit0
workers, driver drain PASS, verified closed archive and exactly two accepted
error-free journal rows before the original physical/capture audit. A real
worker/drain/archive deadline remains PARTIAL; completed episodes cannot mask it.
P2 supplied [P2_to_Q](../REQUESTS/P2_to_Q.md) after its one GPU diagnostic.
Q adopted that exact owned pi_native.py patch: G1 A now uses the independent
R16 eager public-stage reference, with exact normalized/wire/debug/suffix bytes.
K10 stock timing alternate remains exact. K2/K1 whole-call compiled output is
a separate diagnostic, admitted only for P2's sealed checkpoint/norm/archive
profile: absolute max/mean limits .08/.02 and .13/.03, rtol0. These bounds never
relax native parity or imply policy quality. Receipts name the reference path,
parity contract and alternate comparison explicitly. The coordinator's corrected
G1 A retry now PASS (run slot_G1_pi05_mw_A_20261009_191908_601423); retain it and
run G1 B next. The earlier FAIL receipt remains unchanged.

A verifies owner native/staged/normalization/noise/geometry and exact debug/
suffix products. π tests K10/2/1; DP K100/2/1 and explicit DDIM suffix with production B1 only;
Fast K10/2/1; GR K4/2/1 is intentionally three separate immutable-K endpoints.
DP requires status pass and skipped0, with passed equal to the number of actual
passing checks; the count is never hard-coded. Its probe consumes Q's pinned
load descriptor and recorded inputs, preserving protocol/wire-size identity.
P4 GPU fix2 disables production DP B>1 in all three stage capabilities. Q
requires real_production_B1_only, max_batch_size1 and exact B1 queue/direct
bytes. B2 composition measurements remain labelled offline diagnostics and
cannot count as production batching support or parity certificates. Fresh
DP source/model/library seals are required by this release.
GPU tolerance is only the owner's recorded upstream/batch tolerance; debug
parity and native continuation remain exact. Wrong content/norm/model seals
fail. Dedicated C/B2 is GR native S3 only. Both Fast C/B2 wrappers always print
PARTIAL with gpu_executed=false and start no model. R
qualifies two resident models and actual stock sticky proxy hot-bundle control;
throughput/pricing are not inferred.

B requires ready/closed native server, two real WS clients, full/debug byte
parity, two hot bundles, synthetic CP1 HIT (1,0,0)/NFE0, two invalid selections
rejected, exactly two accepted error-free real episodes, attested workers,
contiguous captured decision ledgers and joined actual client controls. Applied
counts must be 1..5, with native protocol caps. Every server execution count,
cursor and final terminal acknowledgment must match the contiguous physical
control ledger. Early success is acceptable;
policy failure after the full control cap is an accepted mechanics outcome.
An absent or contradictory acknowledgment fails the slot. Q never imputes
execution counts. Each loaded bundle binds its sealed store, teacher sampler,
recipe and task vocabulary through P1's R17 server and recipe factory.
Reports include JSON/CSV/HTML/PNG and no full500 success-rate claim.

G3 collects 3×3 actual native GPU decisions on fixed inputs (synthetic MW,
recorded paired256 LIBERO), checks byte parity/native suffix/native reset when available, closes
NPZ→exact H5, builds a fixed-input library, performs library-only AWM/guard/
LOEO corrector/periodic fit, reloads numeric fit and compares action/donor/
weight/diagnostic replay, loads a new stock artifact for an actual forced HIT,
freezes dependencies and audits accepted identities/errors with four formats.
**Every G3 slot returns PARTIAL even if all these checks pass:** stock native
warm/reset dispatch, certified CP2/twins/shadow/routing combinations, real
dev/test freeze/cohorts and semantic physical restore still need owner work.
The P6 hook's absent callable has been filled for the implemented subset only.

The G1 C tail slot likewise returns PARTIAL: its actual model cursor byte proof
uses synthetic one-control acknowledgments. A distinct real replan1 physical
protocol remains P2's required handoff. Formal H5/replan5 commit stays 5.
GR fixed-K native A_k2/A_k1 have passed on GPU. GR S1/S2 production batching
remains outside the qualified scope.

Requests: [Q_to_P1](../REQUESTS/Q_to_P1.md),
[Q_to_P2](../REQUESTS/Q_to_P2.md), [Q_to_P4](../REQUESTS/Q_to_P4.md), [Q_to_P6](../REQUESTS/Q_to_P6.md).
BF16 observer transport is repaired at the Q boundary with lossless float32
values plus original uint16 bits, leaving model/RNG/resume tensors untouched.
It is CPU-tested. P6 now implements lossless native BF16 encoding upstream;
the final payload includes that fix and Q retains its tested boundary.
'''
    fix2=q/'receipts/GPU_FIX4.json'
    if not fix2.is_file():fix2=q/'receipts/GPU_FIX3.json'
    if not fix2.is_file():fix2=q/'receipts/GPU_FIX2.json'
    if fix2.is_file():
        release=json.loads(fix2.read_text())
        generation={'r17.Q.gpu_fix4.v1':'GPU fix4','r17.Q.gpu_fix3.v1':'GPU fix3'}.get(release.get('schema'),'GPU fix2')
        text+=f'\n## {generation} 4090 ready execution order\n\n'
        text+='Run serially from the coordinator host when GPU admission allows; preserve the running evals. Stop dependent branches when their required A/B fails. Preserve already-passed π A, GR A/current+60000/K2/K1 and DP-LIBERO A+B.\n\n'
        text+='| Order | Command | Readiness / prerequisite |\n|---:|---|---|\n'
        for i,row in enumerate(release['local_queue'],1):
            text+=f"| {i} | `bash exp/r17_integration/qualification/{row['slot']}.sh` | {row['prerequisite']} |\n"
        text+=f"\nThe first {len(release['ready_now'])} rows are ready now. Remaining rows depend on the new PASS receipts named in the table. G3/tail retain their intentional PARTIAL scopes.\n\n"
        text+=f'## {generation} H100 waiting order\n\nAll GPU rows wait for both trainings to exit, approximately22:15 CDT at the later current estimate. No qualification overlap with either training is authorized.\n\n'
        for i,row in enumerate(release['h100_queue'],1):
            text+=f"{i}. `{row['slot']}.sh` — {row['prerequisite']}\n"
        text+=f'\nDP current A/B names are compatibility aliases of formal final100000 A/B; do not duplicate a passed identical checkpoint. Fast final16001 remains gated on immutable publication and CPU sealing. Both Fast C/B2 commands are retired CPU-only. See [release index](receipts/{fix2.name}) and [handback](HANDBACK_Q.md).\n'
    (q/'PLAN.md').write_text(text)
    print('PASS PLAN.md: all 34 slots ordered')


if __name__=='__main__': main()
