# R12 opus — offline evidence for "look half" (one camera per look)

opus, 2026-10-03 CDT. Library-only (B-pool `bpool_cs` / `bpool_all` and R10 nested subsets) plus the allowed non-test
closed-loop material (R6 B-val recordings `r06_c_cal`, R11 dev decision logs). Nothing under
`offline_search_store/queries` and no test-A closed-loop root was opened. All scripts are in `tools/`; outputs in
`out/`. Prefix for every command:
`taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r12.opus.tools.<script>`

> **Read §12 first.** After DIRECTIVE_2 (owner, 12:5x CDT) prices are compute shares (one camera = half the vision
> price) and latency arguments do not count: §0 items 1 and 7 and §2's price discussion are withdrawn. §12 has the
> closed-loop dev evidence; its main finding is that the guard, not retrieval quality, limits look-half.

## 0. Summary

1. **The price being saved is mostly an artifact of how pi0.5 encodes cameras.** `PI0Pytorch.embed_prefix` embeds
   the three image slots one after another (the third slot is an always-masked dummy). The vision tower is
   launch-bound at batch 1: one image 23.0 ms, two images in one call 23.5 ms, three 22.9 ms; sequential two 46.8 ms,
   three 69.0 ms (local 4090, tower-only microbenchmark, §2). If both real cameras are embedded in one call, a
   two-camera look costs about what a wrist-only look costs, so look-half has almost nothing left to save, while
   every look and every policy call gets ~.08 of a call cheaper (≈ −.04 IR per decision slot) at bf16-level key
   changes (0.15 %). GR00T's stage 1 already embeds both images in one `extract_feature` call, so a GR00T one-camera
   look probably costs the same as a two-camera look. **This must be measured on the serving path (REQUESTS 0)
   before any look-half arm is built.**
2. **One camera is not as good as two, and the best single camera is the wrist.** Library k-fold replay (§3): the
   wrist-only cache (own refitted metric) serves chunks 0.010–0.025 further from the policy's own chunk (motion RMS,
   σ units; +2–6 %) than the two-camera cache; third-person-only is worse on pi0.5 and similar on GR00T; robot state
   alone is surprisingly close offline, which shows that library replay under-states what vision does in closed loop.
3. **Nothing made one camera as good as two at every look:** refitted metric hyperparameters, a 128-d wrist PCA,
   other poolings of the stored wrist tokens (2×2, 8×8, global, normalised), a continuity term, and
   lower-resolution re-encodes of the raw images all failed (§3–4).
4. **The agentview's information is slow:** the wrist view plus the agentview *from the previous look* (10 controls
   old; library features lagged identically) is almost as good as two fresh cameras (+0.000–0.016). Older agentview
   (20–40 controls) loses this.
5. **Where is one camera safe?** Not in "easy kinematic stages": R7's gripper-stage signals predict nothing
   (AUC 0.44–0.66). What predicts it is a post-look self-consistency test: *does adding the previous look's agentview
   change the wrist-only answer?* (AUC 0.77–0.87 for large deviations from the two-camera answer). Safe looks are
   front-loaded (first fifth of an episode 66–82 % safe vs last fifth 33–48 % on 50-episode libraries) and rarer in
   failing episodes (§6).
6. **A wrist-first cascade is the only safe form** (§7): encode the wrist camera, test consistency, complete the
   agentview only when the test fails. With a library-size tolerance (0.15 at 50 episodes, 0.075 at 200, 0.05 at 500)
   it serves 42–49 % (50 eps) / 20–34 % (200) / 8–16 % (500) of looks one-camera at a predicted SR change of
   −0.1 to −0.3 pp. A pre-look rule with the same share doubles the excess error. **Post-freeze dev check (§11):** on
   closed-loop dev looks the frozen τ lets through ~3× more retrieval cost; the dev-calibrated τ' = .10·√(50/N) keeps it
   at +.0014 with a 40 % one-camera share on the 50-episode cells.
7. **Savings are small once priced honestly** (§7): under the owner price table the cascade saves .019–.031 IR per
   slot on pi0.5, but 60–85 % of that is "wrist + completion (.115) is cheaper than a stock full look (.152)", i.e.
   the masked third slot again. With a full look priced at .115 the cascade saves .002–.012. Aggressive look-half
   (wrist-only at every look) saves .009–.024 (fair) for a predicted 0.5–1.5 pp SR loss: not worth it at matched SR.
8. **Closed-loop check** (§8): on cache-driven R6 B-val states the consistency test passes as often as in library
   replay (500-episode cells: 0.53 vs 0.54 at tolerance .1), and the one-camera answers it lets through differ from
   two-camera as little (dF 0.07–0.11).

## 1. Setup

- **Replay** (`replay.py`, `replay2.py`, `replay3.py`): R10's nested subsets (50/200/500 episodes), 5 folds by
  permutation position % 5 (50 eps = leave-one-episode-out). PCA-64 per camera refitted on the training fold (sizes
  < 500) or the stored r01 bpool basis (500), per-task AWM Mahalanobis metric (`fit_metric`, nn 3, λ .1, main +
  early) refitted **per feature set** on training rows, top-16, kref 5 (50 eps) / 8, kernel mean of library chunks,
  regime 2 (after a HIT). Every held-out row is a query; "looks" are the even steps ≥ 2 (deployed cadence).
- **Yardstick** e10 = RMS over 10 steps × 6 motion channels of (served − the row's own stored policy chunk)/σ, σ the
  parent library's action std; `dF` = the same RMS between a variant's served chunk and the two-camera one. Uncorrected
  chunks unless stated.
- **Feature sets**: F = [PCA64 v0, PCA64 v1, rs8] (deployed); W = [v1, rs] (wrist); A = [v0, rs]; R = [rs];
  WL = [v0 from two decisions earlier, v1, rs] with the library side lagged the same way; WL4/6/8 = lag 4/6/8.
  `key_v0` = agentview (`base_0_rgb` / `video.image`), `key_v1` = wrist (`left_wrist_0_rgb` /
  `video.wrist_image`); verified on images (`out/views_row20.png`) and in the GR00T key builder.

## 2. Costs and the ceiling

**Which looks could be half looks** (`dev_look_reasons.py`, R11 dev logs, B-pool inits): looks are exactly two
decisions apart (v = .502–.514). Knob off: 66–84 % of looks are regular looks, the rest are the look right after a
policy call (lifecycle) and step 0. With the periodic knob at .32, regular looks fall to 41–44 % and most cache looks
are the lifecycle looks after a call. A look-half rule restricted to regular looks (R7's plugin rule) therefore loses
most of its share when the knob is on; my design allows the post-call look too.

**Owner price table** (R7 Addendum A): full .152 (stock) / .1656 (per-request path), wrist .0646, completion .0502.
Saving per one-camera look that stays a HIT: .0874; that turns into a call (completion needed): .037. Ceiling on
pi0.5 knob off (every eligible look one-camera): ≈ v·(.0874·(1−g) + .037·g) ≈ .035–.04 IR per slot.

**Where the prices come from** (`bench_tower.py`, `out/bench_tower.json`; HF SigLIP-so400m + projector rebuilt from
`pi05_libero_pytorch/model.safetensors`, bf16 eager, idle local 4090; validated against stored library tokens:
relative key error 0.15–0.22 %, cosine 0.999998):

| tower call | 1 image | 2 | 3 | 4 | 8 | 16 | 2 sequential | 3 sequential |
|---|---|---|---|---|---|---|---|---|
| median ms | 23.0 | 23.5 | 22.9 | 25.3 | 25.0 | 34.3 | 46.8 | 69.0 |

Resolution does not matter either (112 / 168 / 224 px all ≈ 22 ms). R7's stock stage 1 (62.1 ms) is three
sequential embeds; wrist-only (26.4 ms) is one embed plus the cached dummy; completion (20.5 ms) one embed. A
two-camera look with both cameras in one call would cost ≈ 26–29 ms ≈ the wrist-only look. Batched keys differ from
per-camera keys by 0.15 % relative L2 (`batch_parity.py`), the same as recomputation noise vs the stored library.
GR00T: `GrootStagedRunner.run_stage1` calls `extract_feature(pixel_values)` once with both images, so its
one-camera look is probably priced like its two-camera look (not measured).

## 3. One camera vs two (library replay, uncorrected; `analyze_replay.py` → `out/replay_tables.txt`)

Paired Δe10 vs F on look rows (F's e10 in brackets):

| cell-size | F e10 | W | A | R | WL (lag 2) | WL4 | WL8 | ref Fl3 | ref Fpca |
|---|---|---|---|---|---|---|---|---|---|
| pi05 l10 50 | .451 | +.013 | +.037 | +.048 | +.001 | +.021 | +.032 | −.008 | −.005 |
| pi05 spatial 50 | .533 | +.014 | +.023 | +.013 | +.017 | +.010 | +.014 | −.009 | +.003 |
| groot l10 50 | .498 | +.023 | +.025 | +.038 | +.016 | +.029 | +.043 | −.002 | +.008 |
| groot spatial 50 | .530 | +.010 | +.004 | −.028 | +.002 | +.004 | +.011 | −.013 | −.010 |
| pi05 l10 200 | .337 | +.019 | +.044 | +.055 | +.010 | +.019 | +.026 | .000 | .000 |
| groot l10 200 | .433 | +.016 | +.024 | +.018 | +.000 | — | — | — | — |
| pi05 l10 500 | .306 | +.019 | +.050 | +.057 | +.010 | — | — | — | — |
| groot l10 500 | .392 | +.016 | +.034 | +.015 | +.006 | — | — | — | — |
| pi05 spatial 500 | .355 | +.015 | +.048 | +.049 | +.007 | — | — | — | — |
| groot spatial 500 | .402 | +.025 | +.021 | +.006 | +.007 | — | — | — | — |

(Fl3 = F with ridge .3, Fpca = F on another valid PCA basis: two equally reasonable two-camera caches, for scale.
Gripper-sign mismatch: W equals F everywhere; A and stale-agent variants are worse on pi0.5 L10.)

- **Wrist is the better single camera on pi0.5** (A is 2–3× worse); on GR00T the two are close (A better on Spatial).
- **State alone (R) is close offline**, even better than F on GR00T Spatial-50. R7/R9 showed that removing vision
  costs success in closed loop (GR00T skip-look −9 to −15 pp), so pure-policy replay under-states vision: on a policy
  rollout the arm state already "knows" where the object is. Offline parity of a one-camera variant is therefore
  necessary, not sufficient.
- **Agreement with F** (top-1 identical / kernel-mass overlap / dF): W .38–.51 / .72–.94 / .12–.19; the two reference
  caches .61–.89 / .94–.99 / .03–.12.
- **Tried and failed to close the gap** (Δe10 vs F for pi05 l10 50 / groot l10 50 / pi05 l10 500): wrist metric with
  ridge .3: +.015/+.027/+.021; ridge 1: +.018/+.036/+.023; 8 action-neighbour pairs: +.014/+.029/+.019;
  W128 (128-d wrist PCA): +.014/+.028/+.019; continuity term to the previous kernel's continuation (AWM regime-1
  form): +.049/+.063/+.056 (also hurts F); imputing the agentview from the previous kernel's members advanced two
  rows (WI): +.019/+.014/— (`replay3.py`, `replay2.py`).
- **Stored-token poolings of the wrist view** (`tok_repr.py`, 50-episode L10): 4×4 (deployed) +.013/+.023, 8×8
  +.013/+.026, 2×2 +.013/+.028, global mean +.022/+.038, normalised 4×4 +.012/+.022 (pi05/groot). No gain.
- **Raw-image re-encodes at lower resolution** (`lowres.py`, pi0.5 only, both cameras through the rebuilt tower):
  both cameras at 112 px +.027/+.011/+.028 (L10-50 / Sp-50 / L10-200); at 168 px +.003/+.007/+.013; full wrist +
  168-px agentview −.005/+.002/+.001. A coarse agentview is enough, but at batch 1 a low-resolution encode costs the
  same time as a full one (§2), so it saves nothing.

## 4. The guard under one camera

The frozen only-no-progress guard fires when the served top-1's library progress fails to advance by > .5 steps
between consecutive looks. Its flags follow top-1 identity, and even the two reference caches agree with F's flags on
only 83–98 % of looks. W agrees on 76–94 %, with a similar rate (W g − F g = −.002 to +.025). Design consequence: a
one-camera look uses its own top-1 for the guard ("own guard", call rate unchanged within ±.02 in simulation, §7);
completing the agentview and re-deciding on F after every one-camera flag removes guard calls that F would have made
(g .225 vs .268 on pi05 L10-50) and is not used.

## 5. The layer-3 corrector on one-camera looks (`corrector_check.py`)

Per fold and task a LOEO head (R10 recipe: 384 RFF, ridge 100, episode weights, blend .5) is fitted; Δe10 vs the
two-camera corrected chunk:

| cell-size | W, no correction | W + deployed head, true agentview (oracle) | stale agentview | imputed agentview | **wrist-only head** |
|---|---|---|---|---|---|
| pi05 l10 50 | +.015 | +.011 | +.015 | +.018 | **+.014** |
| pi05 spatial 50 | +.021 | +.012 | +.018 | +.021 | **+.015** |
| pi05 l10 200 | +.028 | +.017 | +.024 | +.022 | **+.018** |
| groot l10 50 | +.021 | +.023 | +.028 | +.027 | **+.021** |
| groot spatial 50 | +.012 | +.012 | +.019 | +.015 | **+.010** |
| groot l10 200 | +.015 | +.015 | +.022 | +.017 | **+.014** |

Feeding the two-camera head a guessed agentview is useless or harmful; a head trained on the wrist path itself is
the best usable option everywhere and never harmful. The SPEC uses it (task-indexed, so it is layer-5 state like the
existing heads).

## 6. Where is one camera safe? (segmentation by look-half safety)

Target: the one-camera answer deviates strongly from two-camera (dF above its 75th percentile) or is clearly worse
(e10(W) − e10(F) > .05). ROC AUC of signals available at the look (`doubt_auc.py`, `out/doubt_auc_W.json`):

| signal | big deviation | clearly worse |
|---|---|---|
| **|W − WL|** (does the previous look's agentview change the wrist-only answer?) | **.77–.87** | **.64–.73** |
| W vs continuation of the previous kernel | .68–.73 | .59–.66 |
| W's nearest distance / kernel spread | .59–.79 | .55–.65 |
| R7 stage signals: neighbours near a gripper event / gripper-vote ambiguity | .44–.66 | .41–.56 |

R7's stage gate (`sw` rule replayed) only picks 15–24 % of looks (50 eps) and its picks are not safer per look.
**Safety profile** of the consistency test (`safety_profile.py`, tolerance .15 at 50 eps, .047 at 500): share of looks
that pass, by fifth of the episode — pi05 L10-50 .72/.59/.52/.54/.46, groot L10-50 .66/.51/.50/.40/.33,
pi05 Sp-50 .67/.57/.50/.46/.40, groot Sp-50 .82/.64/.41/.53/.48; failed-rollout rows .14–.42; no consistent gripper
open/closed or near-event pattern. Excess error of W on passing looks ≈ 0 (−.000 to +.007) vs +.016 to +.039 on
failing looks. Reading: the agentview matters most later in long tasks (which sub-goal is done) and off-track.

**Pre-look vs post-look** (`prelook.py`): deciding from the previous look's test result (persistence AUC .65–.72)
gives the same share but about twice the excess error (e.g. groot L10-50 at .1: .0046 vs .0016). The test must run
after the wrist look, i.e. completion-for-retrieval is needed.

## 7. Sequential simulation of the cascade (`sim2.py`, `sim3.py`; `out/sim3_all.json`, `out/prediction_table.json`)

Each held-out episode is walked look by look. Look 0 full. Then: encode the wrist; compute W and WL_L (L = decisions
since the last fresh agentview key, ≤ 8, lag-matched metric); score = |W − WL_L| (motion RMS, σ) + first-5-step
gripper-sign disagreement; serve W if score ≤ τ (own guard; a flag completes for the policy call), otherwise complete
and serve F with the deployed guard. Prices: owner (.152 / .0646 / .0502) and "fair" (full .1148 = wrist +
completion). Knob: periodic gap cap in looks (K3 ≈ R11's .25–.32 targets). `dq` = mean over looks of the
retrieval-quality cost (e10 of the one-camera chunk − e10 of the two-camera chunk on looks served one-camera).

**Tolerance rule (library-only, frozen):** per library size the largest τ whose pooled (4 cells) knob-off dq ≤ .002
(≈ 0.2 pp predicted SR, §9): 50 eps → .15 (pooled dq .0019), 200 → .075 (.0020), 500 → .05 (.0009; .075 gives .0021).
These follow τ(N) = .15·√(50/N).

Selected rows, knob off (K0) and K3 (pi0.5 owner/fair ΔIR per slot; GR00T with placeholder half = completion = .074):

| cell | τ | one-camera share | completion share | ΔIR owner | ΔIR fair | Δg | dq | all-wrist: ΔIR owner / fair, dq |
|---|---|---|---|---|---|---|---|---|
| pi05 l10 50 K0 | .15 | .42 | .58 | −.031 | −.012 | −.003 | +.0012 | −.032 / −.013, +.0066 |
| pi05 l10 50 K3 | .15 | .32 | .44 | −.025 | −.011 | −.006 | −.0003 | −.027 / −.013, +.0047 |
| pi05 spatial 50 K0 | .15 | .45 | .55 | −.029 | −.011 | +.002 | +.0027 | −.041 / −.022, +.0081 |
| pi05 l10 200 K0 | .075 | .33 | .67 | −.024 | −.005 | +.007 | +.0017 | −.033 / −.014, +.0148 |
| pi05 l10 500 K0 | .05 | .16 | .84 | −.021 | −.002 | +.005 | +.0007 | −.028 / −.009, +.0134 |
| pi05 spatial 500 K0 | .05 | .15 | .85 | −.023 | −.004 | +.001 | +.0013 | −.039 / −.020, +.0142 |
| groot l10 50 K0 | .15 | .38 | .62 | −.019* | −.019* | −.010 | +.0019 | −.022*, +.0121 |
| groot spatial 50 K0 | .15 | .49 | .51 | −.031* | −.031* | −.028 | +.0018 | −.030*, +.0046 |
| groot l10 500 K0 | .05 | .08 | .92 | −.002* | −.002* | +.002 | +.0005 | −.019*, +.0109 |

\* GR00T prices are placeholders; per §2 the real one-camera saving is probably ≈ 0.

The owner-basis saving of the cascade is mostly the cheaper completion path (−.037 per look × v ≈ −.019 per slot on
every completed look), not the one-camera looks themselves (−.050 × share × v ≈ −.004 to −.011).

## 8. Closed-loop check on R6 B-val states (`cl_agreement.py`, `lag2_share.py`)

80 pure-cache episodes (1 B-val init per task, 8 cells, 112–355 looks per cell, both camera keys at every look) were
re-served by the R10 subset library of the same cell (B-val init's own library episode excluded). Lag-2 consistency
test, share passing at τ = .1 / .15, closed loop vs library replay:

| cell | closed loop | library | dF of passing looks (CL) |
|---|---|---|---|
| pi05 l10 500 | .53 / .72 | .54 / .73 | .072 / .088 |
| pi05 spatial 500 | .54 / .72 | .54 / .77 | .080 / .091 |
| groot l10 500 | .39 / .58 | .39 / .60 | .095 / .107 |
| groot spatial 500 | .46 / .70 | .43 / .67 | .089 / .104 |
| pi05 l10 50 | .29 / .49 | .39 / .56 | .097 / .117 |
| groot l10 50 | .19 / .31 | .29 / .47 | .097 / .112 |

On the 500 cells the recording library equals the evaluation library and the statistics match. The 50-cell
recordings were driven by the curated `current` library, so their states sit further from the R10 subset; the share
is lower there, i.e. the test fails safe. Failing looks deviate 2–3× more (dF .15–.47).

## 9. From offline error to success rate (calibration used for predictions)

There is no clean offline→SR map; R10's corrector moved offline e10 by ~0–5 % yet gained 2–4 pp on L10. I use the
cross-library-size relation of the same system (R10 aggregates: SR vs my replay e10 at 50/200/500): 0.5–2 pp SR per
.01 e10 (central 1 pp). Consistency check: R7's stage-wrist arm, replayed here (`sw`), gives dq ≈ +.0002 to +.0069 →
predicted −0.0 to −0.7 pp; R7 measured −0.05 [−1.40, +1.28] pp pooled. Knob conversion for matched SR: R11 periodic
≈ 2.9 pp per +.1 IR on 50-episode cells, near zero on near-pure dense cells.

## 10. Files

`tools/common.py` (folds, PCA cache, metric, stage table), `replay.py`/`replay2.py`/`replay3.py` (replays),
`analyze_replay.py`, `doubt_auc.py`, `cascade_sim.py`/`sim2.py`/`sim3.py` (simulations), `prelook.py`,
`safety_profile.py`, `corrector_check.py`, `cl_agreement.py`, `lag2_share.py`, `dev_look_reasons.py`, `tok_repr.py`,
`lowres.py`, `bench_tower.py`, `batch_parity.py` (GPU, < 4 GB). `cache/` holds disposable PCA projections.

## 11. Post-freeze validation on the coordinator's B-pool dev recordings (read after PREDICTION was frozen)

`dev_validate.py` replays the frozen cascade decision look by look on `r12_devlooks_50L` / `50S` (deployed five-layer
system, knob off, two-camera serving, 100 dev episodes per arm, both keys at every look; L10-50 also with the policy's
shadow chunk). Library side as SPEC 2.1 (deployed artifact PCA, metrics refitted on the served rows; my F replica
matches the deployed top-1 on 100 % of regime-2 HIT looks).

| arm | τ | one-camera share | dF one-camera / completed | retrieval cost vs shadow policy |
|---|---|---|---|---|
| pi05 L10-50 | .15 (frozen) | .55 | .101 / .240 | +.0035 ± .0009 |
| pi05 L10-50 | .10 | .40 | .083 | +.0014 ± .0006 |
| pi05 L10-50 | .075 | .29 | .071 | +.0010 ± .0004 |
| pi05 L10-50 | .05 | .16 | .061 | +.0000 ± .0003 |
| pi05 L10-50 | wrist at every look | 1.00 | — | +.0057 ± .0023 |
| pi05 Sp-50 | .15 / .10 / .075 | .54 / .40 / .27 | .099 / .089 / .076 | no shadow chunks |
| groot Sp-50 | .15 / .10 / .075 | .60 / .40 / .28 | .119 / .099 / .087 | no shadow chunks |

On cache-driven states the test passes more often than in library replay and its selectivity is lower: the frozen
τ = .15 lets through a closed-loop retrieval cost about 3× the offline value. The all-wrist cost matches offline.
Dev-calibrated tolerance τ'(N) = .10·√(50/N) (SPEC §5, PREDICTION Addendum A). Coordinator's measured prices
(`prep/HANDBACK.md`): pi0.5 wrist .0670, third-person .0666, completion after wrist .0512; GR00T one-camera
.143–.150 vs full .148 (GR00T look-half has no price to save, as predicted from the code).

---

## 12. Directive 2 (owner ruling 12:5x CDT): compute-share prices, look-half as a cache-layer method

**Prices now:** π0.5 full look .152 = one camera .076 + completion .076; GR00T .148 = .074 + .074; a policy call adds
.848 / .852. Proposal S (batched camera encode) and every latency/launch argument in §0–§2 are withdrawn: they are
implementation speed-ups, not methods. A completed one-camera look now costs exactly a full look, so the cascade
loses nothing by completing; each look kept one-camera saves .076·v ≈ .038 IR per slot-share.

### 12.1 Closed-loop dev feature tables (`dev_features.py` → `out/dev/*.npz`)
All B-pool dev recordings: π0.5 + GR00T L10-50 (shadow policy chunks), π0.5 + GR00T Sp-50, π0.5 L10-200 / Sp-200,
π0.5 + GR00T L10-400, and the periodic-knob (.32) L10-50 arms of both models; 100 episodes per arm (knob arms 50).
Every look re-served by the arm's library with F, W, A, WL2/4/6/8 (fresh wrist + agentview remembered from 2–8
decisions earlier), AL2 (fresh agentview + remembered wrist); my F replica reproduces the deployed top-1 on 100 % of
regime-2 HIT looks and the deployed guard decision on 100 % of looks (flag ⇔ recorded call).

### 12.2 Retrieval cost alone (`dev_eval.py`; calls kept where they were recorded)
pi05 / groot L10-50, retrieval cost vs the shadow policy chunk (mean over looks):

| schedule | one-camera share | retrieval cost |
|---|---|---|
| wrist only at every non-call look | .76 / .67 | +.0083 / +.0084 |
| alternate the encoded camera, other camera remembered from the previous look | .76 / .67 | +.0066 / +.0013 |
| cascade τ .10 (serve wrist) | .34 / .23 | +.0013 / +.0010 |
| cascade τ .10 (serve wrist + remembered agentview) | .34 / .23 | +.0013 / +.0006 |

Camera memory helps (strongly for GR00T). On retrieval quality alone look-half looks cheap.

### 12.3 The guard is the binding constraint (`dev_eval2.py`, `call_predict.py`, `call_predict2.py`, `stall_cues.py`)
On cache-driven states the frozen only-no-progress rule evaluated on the one-camera top-1 fires on only 64–72 % of
the looks where the two-camera guard called, and spuriously on 10–19 % of the others. With "complete and let the
deployed guard decide on F" on every one-camera flag, the one-camera system still **misses 28 % (π0.5) / 28 %
(GR00T) of the guard's calls with the wrist only, 43 % with camera alternation, 9.5 % / 5 % with the cascade at
τ .1**. A missed call executes the cache instead of the policy at a stall: on L10-50 this dominates the cost
(cascade τ .1: retrieval-only +.0013, with missed calls +.0106). R9 saw the same in its own closed loop (GR00T
wrist-only: first guard call ~5 steps later, −9 to −15 pp).
- Calls are retrieval-progress stalls, not physical stalls: robot-state motion predicts them at AUC .36–.53. The best
  one-camera cue is the smaller progress advance of the wrist and wrist+memory top-1 (AUC .82–.93).
- To keep ≥ 97 % of calls, a one-camera look must be completed whenever (previous look called) or (that advance
  ≤ .5 + m library steps, m ≈ 1–2). A learned logistic predictor (leave-one-arm-out) is no better than this rule
  (98 % recall keeps 20–25 % of non-call looks on L10, ~60 % on Spatial).

### 12.4 The guarded method on dev (`dev_eval3.py`, `out/dev_M1_grid.json`; τ .10, serve wrist + memory)
| arm (knob off unless noted) | m | one-camera share | ΔIR (vision / missed calls) | missed share of calls | retrieval cost (shadow) |
|---|---|---|---|---|---|
| pi05 L10-50 | 1.0 | .23 | −.0088 / −.0026 | .023 | +.0023 ± .0007 |
| pi05 L10-50 | 1.5 | .13 | −.0048 / −.0011 | .009 | +.0008 ± .0005 |
| groot L10-50 | 1.0 | .12 | −.0047 / −.0027 | .018 | +.0031 ± .0009 |
| groot L10-50 | 1.5 | .08 | −.0028 / −.0016 | .010 | +.0018 ± .0007 |
| pi05 Sp-50 | 1.0 | .30 | −.0113 / −.0012 | .014 | — |
| groot Sp-50 | 1.0 | .31 | −.0116 / −.0033 | .051 | — |
| pi05 L10-200 | 1.0 | .33 | −.0125 / −.0050 | .069 | — |
| pi05 Sp-200 | 1.0 | .45 | −.0171 / −.0009 | .029 | — |
| pi05 L10-400 | 1.0 | .33 | −.0127 / −.0052 | .086 | — |
| groot L10-400 | 1.0 | .16 | −.0058 / −.0044 | .035 | — |
| pi05 L10-50, knob .32 | 1.0 | .01 | −.0004 / −.0018 | .007 | — |
| groot L10-50, knob .32 | 1.0 | .01 | −.0002 / −.0013 | .005 | — |

With the knob on, 54–57 % of looks are calls and most of the rest follow a call, so nothing is left to look at with
one camera. Where shadow chunks exist, the cost is ≈ missed-call rate × .45 + a small retrieval part.

### 12.5 Matched-SR reading (1 pp SR per .01 cost; knob 2.9 pp per .1 IR on 50-episode cells, ~1 pp per .1 dense)
pi05 L10-50: −.011 IR for −0.23 pp → +.008 IR to buy back → **net −.003**; groot L10-50: net **+.002 to +.004**;
pi05 Sp-50: about **−.008**; dense L10 cells: positive (worse), because the knob buys almost nothing there. Knob on:
≈ 0. Look-half does not give a material matched-SR saving on this system; its best case is a few thousandths of IR
on π0.5 50-episode cells with the knob off.

### 12.6 GR00T camera encoded alone (`groot_mismatch.py`, local 4090, < 7 GB)
200 B-pool LIBERO images (π0.5 library RGB through the GR00T transforms): a camera encoded alone differs from its
slice of the joint encode by 1.7 % (agentview) / 2.1 % (wrist) relative L2; in the library's PCA space the shift is
4 % of the nearest-neighbour distance; top-16 neighbour overlap .988, top-1 unchanged .96–.97 (cross-GPU numerics
gave .936 at step 0 with no SR change). Not a systematic offset (mean-offset correction does not help). Treat as
numerical noise; no library re-encode needed. The dev replay used joint keys, so GR00T dev numbers exclude it.

### 12.7 Corrector on one-camera looks (`corrector_check.py`, wrist + memory chunk)
Serving the wrist + remembered-agentview chunk with the wrist-only head: +.0009 (pi05 L10-50), +.0099 (L10-200),
+.0098 (groot L10-50), +.018 (pi05 Sp-50) vs the two-camera corrected chunk — the best non-oracle option on L10;
feeding the deployed head the remembered agentview is worse or equal.
