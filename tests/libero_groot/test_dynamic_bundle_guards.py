"""Hot-swapped cache bundles on the GR00T LIBERO server.

Configuration identity is normally carried by the process -- one cell, one
server -- and ``--allow-dynamic-bundles`` trades that away so a conductor can
drive many arms through one process. These tests pin what has to hold for that
trade to be safe: the default stays closed, the GR00T guards re-run on the
*loaded* config rather than only on the startup one, and the gigabyte-scale
storage the server already built is read rather than rebuilt.

No GPU and no ``gr00t`` package: the heavy imports in ``main`` sit behind
argparse validation, and the resolver is reached directly.
"""

from __future__ import annotations

import dataclasses
import types

import pytest

from exp.libero_groot import serve_groot_libero as sgl


@dataclasses.dataclass
class _FakeBundle:
    cache_config: object
    shared_storage: object
    version: int = 1
    yaml_id: str | None = None


class _FakeConfig:
    """Compared by identity, deliberately.

    ``types.SimpleNamespace`` compares by ``__dict__``, so two structurally
    identical configs are ``==`` equal -- and an assertion written with ``==``
    would then pass even when the guards were handed the startup config instead
    of the loaded one, which is the exact confusion these tests exist to catch.
    """

    def __init__(self, builder_type: str = "cp1_groot_libero_mean") -> None:
        self.key_builder = types.SimpleNamespace(type=builder_type)


def _fake_config(builder_type: str = "cp1_groot_libero_mean"):
    return _FakeConfig(builder_type)


@pytest.fixture
def _no_op_guards(monkeypatch):
    """Neutralise the two guards that need a real CacheConfig, recording calls.

    ``_check_libero_builder`` is left real: it is pure, and it is the guard
    whose failure mode (a three-camera RoboCasa builder rejecting every LIBERO
    observation) is the one a hot swap can actually introduce.
    """
    seen: dict[str, list] = {"validate": [], "identity": []}
    guard = types.ModuleType("openpi.cache.groot.load_guard")
    guard.validate_groot_cache_config = lambda cfg, **kw: seen["validate"].append(cfg)
    guard.validate_artifact_identity = lambda storage, cfg: seen["identity"].append((storage, cfg))
    monkeypatch.setitem(__import__("sys").modules, "openpi.cache.groot.load_guard", guard)
    return seen


def _resolve(bundle_id, *, cli_config=None, cli_storage=None, allow_dynamic=True):
    return sgl._resolve_bundle(
        bundle_id,
        cli_config=cli_config,
        cli_storage=cli_storage,
        allow_dynamic=allow_dynamic,
    )


# -- the default stays closed --------------------------------------------


def test_hot_swap_off_rejects_any_other_bundle_id():
    with pytest.raises(ValueError, match="serves only bundle_id='default'"):
        _resolve("gp_l10_fh20", allow_dynamic=False)


def test_hot_swap_off_serves_the_cli_config_under_default():
    cfg, storage = _resolve("default", cli_config="CFG", cli_storage="ST", allow_dynamic=False)
    assert (cfg, storage) == ("CFG", "ST")


# -- what "default" means once hot swap is on ----------------------------


def test_default_falls_back_to_the_cli_config_when_nothing_is_loaded(monkeypatch):
    """The runner selects "default" before the first stage has been sent."""
    monkeypatch.setattr(
        "openpi.serving.websocket_policy_server.get_current_cache_bundle",
        lambda bundle_id=None: None,
    )
    assert _resolve("default", cli_config="CFG", cli_storage="ST") == ("CFG", "ST")


def test_default_yields_no_cache_on_a_teacher_only_server(monkeypatch):
    monkeypatch.setattr(
        "openpi.serving.websocket_policy_server.get_current_cache_bundle",
        lambda bundle_id=None: None,
    )
    assert _resolve("default") == (None, None)


def test_unknown_bundle_id_is_refused_rather_than_silently_substituted(monkeypatch):
    monkeypatch.setattr(
        "openpi.serving.websocket_policy_server.get_current_cache_bundle",
        lambda bundle_id=None: None,
    )
    with pytest.raises(ValueError, match="load_cache_config must precede"):
        _resolve("gp_l10_fh20", cli_config="CFG", cli_storage="ST")


# -- the guards run on the loaded config, not the startup one ------------


def test_guards_rerun_on_the_loaded_bundle(monkeypatch, _no_op_guards):
    loaded = _fake_config()
    monkeypatch.setattr(
        "openpi.serving.websocket_policy_server.get_current_cache_bundle",
        lambda bundle_id=None: _FakeBundle(cache_config=loaded, shared_storage="LOADED_ST"),
    )
    cfg, storage = _resolve("gp_l10_fh20", cli_config=_fake_config(), cli_storage="CLI_ST")
    assert cfg is loaded
    assert storage == "LOADED_ST"
    # Both guards saw the loaded config -- not the startup one, which is what
    # the startup checks already covered. Identity, not equality: see _FakeConfig.
    assert len(_no_op_guards["validate"]) == 1
    assert _no_op_guards["validate"][0] is loaded
    assert len(_no_op_guards["identity"]) == 1
    seen_storage, seen_cfg = _no_op_guards["identity"][0]
    assert seen_storage == "LOADED_ST"
    assert seen_cfg is loaded


def test_a_hot_swapped_robocasa_builder_is_refused(monkeypatch, _no_op_guards):
    """Three-camera builders assert three image-token runs; LIBERO has two."""
    monkeypatch.setattr(
        "openpi.serving.websocket_policy_server.get_current_cache_bundle",
        lambda bundle_id=None: _FakeBundle(
            cache_config=_fake_config("cp1_groot_robocasa_mean"), shared_storage="ST"
        ),
    )
    with pytest.raises(ValueError, match="not a LIBERO builder"):
        _resolve("gp_l10_fh20")


def test_storage_is_read_from_the_bundle_never_rebuilt(monkeypatch, _no_op_guards):
    """Rebuilding would mean one artifact load per connection per arm."""
    calls = []
    monkeypatch.setattr(
        "openpi.cache.config.build_shared_storage",
        lambda cfg: calls.append(cfg),
    )
    monkeypatch.setattr(
        "openpi.serving.websocket_policy_server.get_current_cache_bundle",
        lambda bundle_id=None: _FakeBundle(cache_config=_fake_config(), shared_storage="ST"),
    )
    _, storage = _resolve("gp_l10_fh20")
    assert storage == "ST"
    assert calls == []


# -- CLI refusals (argparse runs before any heavy import) ----------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--allow-dynamic-bundles"], "requires --concurrent"),
        (
            ["--allow-dynamic-bundles", "--concurrent", "--collect-hdf5", "/tmp/x"],
            "cannot be combined with --collect-hdf5",
        ),
    ],
)
def test_cli_refusals(monkeypatch, capsys, argv, message):
    monkeypatch.setattr("sys.argv", ["serve_groot_libero.py", *argv])
    with pytest.raises(SystemExit):
        sgl.main()
    assert message in capsys.readouterr().err
