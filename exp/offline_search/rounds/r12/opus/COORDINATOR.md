# Coordinator notes to opus (R12)

## 2026-10-03 11:3x CDT — REQUESTS.md received and accepted

**Request 1: closed-loop keys.** I will record on B-pool dev inits disjoint from the served subsets, using the deployed system with unchanged two-camera serving and input logging on.
- **Cells:** π0.5 and GR00T, L10-50 and Sp-50, knob off, 100 episodes each.
- **Knob on:** the periodic knob on the L10-50 cells.
- **Shadow chunks:** policy shadow chunks on a deterministic subsample, if the recording path supports them.

**Size 500 cannot be done as asked.** The 500 libraries contain all 50 B-pool inits of every task, so no unserved B init exists. I will not substitute test A. As the near-dense proxy I will record size **400**, which leaves 10 unserved B inits per task, on L10 for both models (100 episodes each). If you would rather have a different proxy, write it in REQUESTS.md.

sol is finishing the recording support. Locations will be written here when the data lands.

**Request 2: prices.** sol is measuring on the local 4090:
- π0.5 third-person-only;
- GR00T full, wrist-only, third-person-only and completion.

I will also ask for your item 3: a π0.5 full look with the masked dummy camera's embedding cached. Results go to `exp/offline_search/rounds/r12/prep/HANDBACK.md` (you may read that file).
- 2026-10-03 12:1x CDT — **Measured single-camera prices are ready.** Full table, raw latencies and method are in `exp/offline_search/rounds/r12/prep/HANDBACK.md`; you may read it. Local RTX 4090, batch 1, owner basis.
  - **π0.5** (full look = .152):

    | path | price |
    |---|---|
    | wrist-only | .0670 |
    | third-person-only | .0666 |
    | completion after wrist | .0512 |
    | completion after third | .0623 |

  - **GR00T** (full look = .148): a half look saves almost nothing, and a half look followed by a policy call costs more than a full look.

    | path | L10 | Spatial |
    |---|---|---|
    | wrist | .150 | .145 |
    | third | .143 | .145 |
    | completion | .134–.138 | .134–.138 |

    The reason is that GR00T's tower encodes two images at about the cost of one. On the current serving stack, look-half for GR00T has essentially no IR value unless its vision path changes.
  - **Separately encoded GR00T cameras are not numerically identical to the batched full encoding.** Max embedding deltas are about 1–4. A GR00T half-look serving mode would need its own accuracy gate.
- 2026-10-03 12:1x CDT — **Dev recordings are running.** Two-camera serving with per-look recording, B-pool dev inits disjoint from the served subsets; 100 episodes per arm unless noted. Roots under `/home/weiland/trace_runs/os_closed_loop/`:

  | root | arms | recorded |
  |---|---|---|
  | `r12_devlooks_50L` | π0.5 and GR00T L10-50, knob off | keys + tokens + 56-px images + shadow policy chunk at every look |
  | `r12_devlooks_50S` | π0.5 and GR00T Sp-50, knob off | keys + images |
  | `r12_devlooks_200` | π0.5 L10-200 and Sp-200, knob off | keys + images |
  | `r12_devlooks_400` | π0.5 and GR00T L10-400, knob off | keys + images |
  | `r12_devlooks_knob` | π0.5 and GR00T L10-50, periodic knob at .32 | keys + images; 50 episodes each |

  - Look files land in `runs/<arm>/server_<port>/looks/`. Read them with `exp.offline_search.closed_loop.look_recording.read_looks`.
  - I will append here when each root completes.
- 2026-10-03 12:1x CDT — **Your price item 3 is not measured yet** (a π0.5 full look with the masked dummy camera's embedding cached). The component sum from this table is wrist 30.2 ms + completion 23.1 ms ≈ 53.2 ms, against a stock full of 68.5 ms, i.e. about .118. That is an estimate, not a measurement. I will have it measured directly once the local GPU is free of the recording servers; timing under load would be distorted.
- 2026-10-03 12:2x CDT — **`r12_devlooks_50L` is complete and readable.** Readers pass hash checks, `init_pool=B`.
  - π0.5 L10-50: 100 episodes, SR .80, IR .177, 3,198 looks.
  - GR00T L10-50: 100 episodes, SR .74, IR .212, 3,384 looks.
  - Every look has keys, tokens, 56-px images and a shadow policy chunk.
  - Note that shadow inference perturbs nothing served; it is not charged to IR.
  - Path: `/home/weiland/trace_runs/os_closed_loop/r12_devlooks_50L/runs/<arm>/server_<port>/looks/`.
  - Next on the local fleet: `r12_devlooks_400`, then `r12_devlooks_knob`. The H100-served roots (`_50S`, `_200`) are delayed by a mirror-path conflict pending an owner decision; they may be re-recorded on the local fleet instead.
- 2026-10-03 12:4x CDT — **`r12_devlooks_400` is complete.** Keys + 56-px images.

  | arm | episodes | SR | IR | looks |
  |---|---|---|---|---|
  | π0.5 L10-400 | 100 | .97 | .130 | 2,615 |
  | GR00T L10-400 | 100 | .87 | .184 | 2,880 |

  `r12_devlooks_knob` (periodic .32) is recording now.
- 2026-10-03 13:0x CDT — **`r12_devlooks_knob` is complete:** π0.5 and GR00T L10-50, periodic knob .32, 50 episodes each, keys + images.
  - The Spatial-50 and size-200 roots are being re-recorded on the local fleet under new names, `r12_devlooks_50S_local` and `r12_devlooks_200_local`. Use these; ignore `r12_devlooks_50S` and `r12_devlooks_200` (H100 path unresolved).
  - I will append here when they finish.
