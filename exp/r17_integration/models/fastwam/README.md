Fast-WAM uses the shared R17 connection factory, bundle validator and WebSocket
facade. Production is B1; reduced K2/K1 are available on the system profile.
Physical episode and full pricing qualification are still pending. The explicit
LIBERO batch study is now available: `serve --max-batch 8 --batch-wait-ms 5
--batch-ledger PATH`, or the local `formal_launch` shim. See [BATCHING.md](BATCHING.md)
for the measured 5.44x gain over the direct local B1 path, numeric drift and the
requirement for a fresh matching batched control. Default B1 remains unchanged.

Serve descriptors contain `benchmark_specs`, one P2 manifest/asdict document for
each `benchmark_protocols` identity. The loader reconstructs the identity and
checks execution compatibility before touching weights. Export official LIBERO
task maps through P2's CLI; use `--released-reference` for `released_libero`.
For MW use `benchmarks.metaworld.FORMAL_SPEC.manifest()`. Do not replace these
documents with model protocol labels. Template descriptors are incomplete;
populate real documents and content seals before serving. Diagnostic descriptors
can omit documents only with `diagnostic=true` and exclusively `fake_*` identities.

`--debug-directory` attaches P6's capture sink factory to each built P1 core. It
binds the actual sampler and model of that connection/bundle and closes writers
on exit. It is mutually exclusive with the process callback `--capture-sink`.
P2 episode provenance (including task UID/data hash and actual execution ack) is
required for accepted captures; model debug output alone is not a physical or SR
certificate.

Native snapshots now bind CPU-prepared image/state/prompt values. Snapshots from
another observation cannot resume even when the prompt is identical. This is an
internal scope version change; old prompt-only snapshots cannot resume. Generic
warm executor dispatch remains unqualified and fails closed in the P1 framework.

`measure` reports key_ms=prepare+stage1 and hit_path_ms=prepare+stage1+wire, with
their full-path shares. Wire uses the full-output transform as a proxy. Retrieval
search is explicitly excluded and null, never assumed zero. These are isolated
synchronized B1 phase measurements. Formal decision pricing uses P1's host wall
ledger; capture rows are ineligible. CPU fixture timings are not model prices.

`loader_parity --header-only` uses the actual P3 loader and released
`FastWAMPolicy.from_pretrained(strict=True)` constructors on meta tensors. Frozen
sidecars and tensor body loading are substituted. It compares the published
LIBERO geometry, inference config, all unique parameter shapes/dtypes and K10
schedule. This does not establish VAE/T5/tokenizer numerical equivalence.
The published header has 1,620 older block aliases; the current vendor registers
those blocks under MoT layers. The audit checks strict canonical equivalence and
records that original key sets differ. Unmodified released serialization loading
is therefore not certified.

The separate GPU `loader_parity` command uses a sealed `released_libero`
descriptor, real paired256 observations and explicit local released Diffusers
sidecars (`vae/`, `text_encoder/`) and an independent local released UMT5
tokenizer via `--released-tokenizer`. The released comparison uses a qualification
serialization shim that renames only published aliases and calls native strict
`load_state_dict`; its frozen loader and forward math remain unchanged. It runs
raw and released loaders in sequential
child processes, records sidecar/tokenizer/input hashes, and compares tokens,
latent/context/mask,
normalized and wire actions on two B1 observations, K10/seed42. Offline mode
prevents downloads. A tolerance mismatch fails the receipt; it must not be
promoted to parity. See HANDBACK_P3.md for commands and current CPU evidence.

The B500 pure-cache study is admitted through the model-owned content validator
on CPU and resident registration. All8 recipes switch through the same server
and S07 core; ungated HIT/tail NFE is0 and only calibrated stalls call native
K10. See [CACHE_STUDY.md](CACHE_STUDY.md) for exact source pins, CPU receipts,
matched H100 R2/B16/MPS configuration and the coordinator's pending GPU smoke.
