"""T2 exit gates (manual): the registered pi05_robocasa stack vs the ARCHIVED baseline.

Three gates, per the approved plan (§4.2.3):

* **T2-a structural equivalence** — the legacy ``Policy`` is built from an
  INDEPENDENT ``TrainConfig`` transcribed literally from
  ``exp/robocasa365/baselines/serve_robocasa_pi05_ORIGINAL.py`` (local
  ``_Legacy*`` classes in this file — NOT the registered
  ``_RobocasaGroup``/``_RobocasaDataConfig``, which would let a co-drift of
  registry and test pass unnoticed). Their transform chains must match
  pairwise by type AND by dataclass-field values, and norm stats must be
  ``np.array_equal``.
* **T2-b bitwise parity under a shared explicit noise** — the SAME observation
  and the SAME explicit ``noise`` array fed to both stacks must produce
  bit-equal actions (``np.array_equal``, no tolerance). ``noise`` MUST be
  passed explicitly: ``Policy.infer(..., noise=None)`` samples fresh
  flow-matching noise per call. A negative control (different noise => a
  different action) guards against a degenerate always-equal comparison.
* **T2-c collection artifact compliance** — pins the wrapper stack the launch
  command produces; the end-to-end h5 check runs in manual gate 7.

Run (fails — never skips — on a missing checkpoint or GPU once ``--run-manual``
is passed)::

    uv run pytest tests/robocasa365/test_pi05_stack_parity_manual.py --run-manual -q

Evidence lands in ``exp/robocasa365/analysis/t2_parity.txt``. §4.3.4a server
provenance is captured by ``test_t2d_real_server_provenance`` from the RUNNING
collection server — raw metadata off the live WebSocket handshake and the
actual argv from ``/proc/<pid>/cmdline`` — never from constants in this file.
It needs two env vars (the gate FAILS, not skips, without them)::

    ROBOCASA_T2_SERVER=127.0.0.1:8010 \
    ROBOCASA_T2_SERVER_PID=$(pgrep -f "[s]erve_policy.*--port 8010") \
    uv run pytest tests/robocasa365/test_pi05_stack_parity_manual.py --run-manual -q
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.manual

CKPT = pathlib.Path("/home/weiland/ckpt_pi05_robocasa_pytorch")
NS_DIR = CKPT / "assets" / "robocasa"
EVIDENCE = pathlib.Path(__file__).resolve().parents[2] / "exp" / "robocasa365" / "analysis" / "t2_parity.txt"


# ------------------------------------------------------------------
# Legacy config — transcribed from serve_robocasa_pi05_ORIGINAL.py:28-66.
# Deliberately duplicated here instead of importing the registered classes:
# the whole point is an independent baseline the registry can drift AGAINST.
# ------------------------------------------------------------------


def _legacy_train_config():
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.policies import robocasa_policy
    from openpi.training import config as _config

    @dataclasses.dataclass(frozen=True)
    class _LegacyRobocasaGroup:
        """serve_robocasa_pi05_ORIGINAL.py:28-41 (_RobocasaGroup)."""

        def __call__(self, model_config):
            return transforms.Group(
                inputs=[
                    robocasa_policy.RobocasaInputs(
                        action_dim=model_config.action_dim,
                        model_type=model_config.model_type,
                    )
                ],
                outputs=[robocasa_policy.RobocasaOutputs()],
            )

    @dataclasses.dataclass(frozen=True)
    class _LegacyRobocasaDataConfig(_config.SimpleDataConfig):
        """serve_robocasa_pi05_ORIGINAL.py:44-56 (_RobocasaDataConfig)."""

        def create(self, assets_dirs, model_config):
            dc = super().create(assets_dirs, model_config)
            return dataclasses.replace(dc, use_quantile_norm=False)

    # serve_robocasa_pi05_ORIGINAL.py:59-66 — name, model and data verbatim.
    return _config.TrainConfig(
        name="robocasa_infer",
        model=pi0_config.Pi0Config(pi05=True, max_token_len=200),
        data=_LegacyRobocasaDataConfig(data_transforms=_LegacyRobocasaGroup()),
    )


@pytest.fixture(scope="module")
def _gate_environment():
    """Hard preconditions: fail (not skip) so 'gate passed' can't mean 'gate never ran'."""
    if not CKPT.is_dir():
        pytest.fail(f"checkpoint dir missing: {CKPT} — run on weilandserver")
    if not NS_DIR.is_dir():
        pytest.fail(f"norm-stats dir missing: {NS_DIR}")
    import torch

    if not torch.cuda.is_available():
        pytest.fail("CUDA unavailable — T2 parity must run on the GPU the servers use")


@pytest.fixture(scope="module")
def _artifacts(_gate_environment):
    """Build legacy and registry stacks SEQUENTIALLY (peak GPU = one policy).

    The 4090 is shared with the owner's other sessions (~33 GB resident); two
    simultaneously loaded pi0.5 models (~16 GB) would not fit next to them.
    The comparison criterion is unchanged — the SAME observation and the SAME
    explicit noise are fed to both stacks; only the residency overlaps differ,
    and actions are compared as stored arrays. Transform chains and metadata
    are CPU objects and survive the model teardown.
    """
    import gc

    import torch

    from openpi.collect.collection_policy import CollectionPolicy
    from openpi.collect.data_collector import EpisodeDataCollector
    from openpi.policies import policy_config as _pc
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config

    obs = _observation()
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((50, 32)).astype(np.float32)
    other_noise = rng.standard_normal((50, 32)).astype(np.float32)

    ns = _normalize.load(NS_DIR)
    legacy = _pc.create_trained_policy(_legacy_train_config(), CKPT, norm_stats=ns, pytorch_device="cuda")
    legacy_chains = {attr: _flatten_chain(getattr(legacy, attr)) for attr in ("_input_transform", "_output_transform")}
    legacy_meta = legacy.metadata
    a_legacy = np.asarray(legacy.infer(dict(obs), noise=noise)["actions"])
    del legacy
    gc.collect()
    torch.cuda.empty_cache()

    registry = _pc.create_trained_policy(_config.get_config("pi05_robocasa"), CKPT, pytorch_device="cuda")
    registry_chains = {attr: _flatten_chain(getattr(registry, attr)) for attr in ("_input_transform", "_output_transform")}
    registry_meta = registry.metadata
    a_registry = np.asarray(registry.infer(dict(obs), noise=noise)["actions"])
    a_other = np.asarray(registry.infer(dict(obs), noise=other_noise)["actions"])
    # t2c while the registry policy is still alive: the --collect wrapper must
    # find the real model through the chain.
    collect_wrap_ok = (
        CollectionPolicy(registry, EpisodeDataCollector(base_dir="/tmp/t2c_probe"))._inner_model  # noqa: SLF001
        is registry._model  # noqa: SLF001
    )
    del registry
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "ns": ns,
        "legacy_chains": legacy_chains,
        "registry_chains": registry_chains,
        "legacy_meta": legacy_meta,
        "registry_meta": registry_meta,
        "a_legacy": a_legacy,
        "a_registry": a_registry,
        "a_other": a_other,
        "collect_wrap_ok": collect_wrap_ok,
    }


def _observation() -> dict:
    rng = np.random.default_rng(1234)
    frame = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    return {
        "observation/state": rng.standard_normal(16),
        "observation/image": frame,
        "observation/wrist_image": frame[::2, ::2].copy(),
        "observation/right_image": frame[::2, ::2].copy(),
        "prompt": "open the cabinet",
    }


def _flatten_chain(composite) -> list:
    """Unpack CompositeTransform into the ordered list of leaf transforms."""
    leafs = []
    for item in composite.transforms:
        if hasattr(item, "transforms"):
            leafs.extend(_flatten_chain(item))
        else:
            leafs.append(item)
    return leafs


def _assert_field_equal(name: str, a, b) -> None:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.array_equal(np.asarray(a), np.asarray(b)), name
    elif dataclasses.is_dataclass(a) and not isinstance(a, type):
        assert type(a) is type(b), name
        for field in dataclasses.fields(a):
            _assert_field_equal(f"{name}.{field.name}", getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, dict):
        assert set(a) == set(b), name
        for key in a:
            _assert_field_equal(f"{name}[{key}]", a[key], b[key])
    elif callable(a) and not isinstance(a, (str, bytes)):
        # Callables (tokenizers etc.): same concrete type is the comparable unit.
        assert type(a) is type(b), name
    else:
        assert a == b, f"{name}: {a!r} != {b!r}"


def test_t2a_structural_equivalence(_artifacts):
    ns = _artifacts["ns"]
    for attr in ("_input_transform", "_output_transform"):
        l_chain = _artifacts["legacy_chains"][attr]
        r_chain = _artifacts["registry_chains"][attr]
        assert [type(t).__name__ for t in l_chain] == [type(t).__name__ for t in r_chain], attr
        for i, (lt, rt) in enumerate(zip(l_chain, r_chain)):
            assert type(lt) is type(rt)
            if dataclasses.is_dataclass(lt):
                for field in dataclasses.fields(lt):
                    _assert_field_equal(
                        f"{attr}[{i}]({type(lt).__name__}).{field.name}",
                        getattr(lt, field.name),
                        getattr(rt, field.name),
                    )
    # Norm stats resolved via asset_id must equal the explicitly loaded ones.
    from openpi.training import checkpoints as _checkpoints

    resolved = _checkpoints.load_norm_stats(CKPT / "assets", "robocasa")
    assert set(resolved) == set(ns)
    for key in ns:
        for field in dataclasses.fields(ns[key]):
            _assert_field_equal(f"norm_stats[{key}].{field.name}", getattr(ns[key], field.name), getattr(resolved[key], field.name))


def test_t2b_bitwise_parity_with_shared_noise(_artifacts):
    a_legacy = _artifacts["a_legacy"]
    a_registry = _artifacts["a_registry"]
    a_other = _artifacts["a_other"]
    assert a_legacy.shape == a_registry.shape == (50, 12)
    assert np.array_equal(a_legacy, a_registry), (
        f"stacks diverge: max|d|={np.abs(a_legacy - a_registry).max()}"
    )
    # Negative control: without it, array_equal proves nothing (e.g. both
    # stacks degenerating to a constant would also 'pass').
    assert not np.array_equal(a_registry, a_other), "different noise produced identical actions"

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(
        "T2 parity evidence\n"
        "--- comparison (in-process; server provenance is captured separately by t2d) ---\n"
        f"sha256(a_legacy):   {hashlib.sha256(a_legacy.tobytes()).hexdigest()}\n"
        f"sha256(a_registry): {hashlib.sha256(a_registry.tobytes()).hexdigest()}\n"
        f"negative control (different noise differs): {not np.array_equal(a_registry, a_other)}\n"
        "--- in-process policy metadata (NOT server provenance; labeled to avoid confusion) ---\n"
        f"legacy.metadata:   {_artifacts['legacy_meta']!r}\n"
        f"registry.metadata: {_artifacts['registry_meta']!r}\n"
    )


def test_t2c_collect_wrapper_stack(_artifacts):
    """The launch command's wrapper stack: CollectionPolicy directly over Policy.

    Evaluated inside the fixture while the registry policy was still resident
    (_find_inner_model walked the chain and accepted the PI0Pytorch model) —
    exactly what serve_policy.py --collect builds with no cache flags.
    """
    assert _artifacts["collect_wrap_ok"] is True


def _checkpoint_sha() -> str:
    ckpt_file = CKPT / "model.safetensors"
    if not ckpt_file.is_file():
        return "n/a"
    digest = hashlib.sha256()
    with ckpt_file.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def test_t2d_real_server_provenance(_gate_environment):
    """§4.3.4a provenance from the RUNNING server — never from constants.

    * raw metadata: the live WebSocket handshake reply of the actual server
      (``WebsocketClientPolicy.get_server_metadata()``);
    * launch command: the actual argv of the server process, read from
      ``/proc/<pid>/cmdline`` (plus its cwd) — whatever the operator really
      launched, not what a test file claims they launched.

    Missing env vars or an unreadable /proc entry FAIL the gate: provenance
    that cannot be captured must not be silently substituted.
    """
    import os

    server_spec = os.environ.get("ROBOCASA_T2_SERVER", "")
    pid_spec = os.environ.get("ROBOCASA_T2_SERVER_PID", "")
    if not server_spec or not pid_spec:
        pytest.fail(
            "ROBOCASA_T2_SERVER / ROBOCASA_T2_SERVER_PID unset — start the collection "
            "server per §4.2.2 and pass its endpoint + pid; constants must not stand in "
            "for real provenance (§4.3.4a)"
        )
    pid = int(pid_spec.strip())
    proc = pathlib.Path(f"/proc/{pid}")
    if not proc.is_dir():
        pytest.fail(f"server pid {pid} is not alive; cannot capture real provenance")
    argv = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode().strip()
    if "serve_policy" not in argv:
        pytest.fail(f"pid {pid} argv does not look like serve_policy: {argv!r}")
    server_cwd = os.readlink(proc / "cwd")

    host, port = server_spec.rsplit(":", 1)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=host, port=int(port))
    try:
        raw_metadata = client.get_server_metadata()
    finally:
        client.close()

    with EVIDENCE.open("a") as f:
        f.write(
            "--- REAL server provenance (t2d, §4.3.4a) ---\n"
            f"server endpoint: {server_spec}\n"
            f"server pid: {pid}\n"
            f"server argv (/proc/{pid}/cmdline): {argv}\n"
            f"server cwd: {server_cwd}\n"
            f"raw server metadata (WS handshake reply): {raw_metadata!r}\n"
            f"checkpoint path: {CKPT}\n"
            f"checkpoint sha256(model.safetensors): {_checkpoint_sha()}\n"
        )
    assert raw_metadata is not None
