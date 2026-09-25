"""CPU tests for the argv contracts of exp/step_diag/serve_diag_pi05.py and serve_diag_groot.py:
mode / cache-config coupling, full pins the env's K, plain requires --exec-steps, the served
argv gets --non-concurrent and the cache flags, and LIBERO GR00T serves every mode on the
production --concurrent factory with the head's K pinned by --denoising-steps."""

import pytest

from exp.step_diag import envs as E
from exp.step_diag import serve_diag_groot as SG
from exp.step_diag import serve_diag_pi05 as SP

COMMON = ["--arm-id", "x", "--experiment-id", "e", "--diag-out", "/tmp/o"]


def test_pi05_parse_modes():
    args, rest = SP.parse(["--mode", "full", "--env-id", "pi05_rc", *COMMON, "--", "--env", "ROBOCASA", "--port", "1"])
    assert args.exec_steps == E.ENVS["pi05_rc"].k_full and rest[:3] == ["--non-concurrent", "--cache", "--env"]
    args, rest = SP.parse(["--mode", "plain", "--env-id", "pi05_libero_10", "--exec-steps", "2", *COMMON])
    assert args.exec_steps == 2 and rest == ["--non-concurrent", "--cache"]
    args, rest = SP.parse(["--mode", "shadow", "--env-id", "pi05_rc", "--cache-config", "c.yaml", *COMMON, "--", "--port", "2"])
    assert rest == ["--non-concurrent", "--cache_config=c.yaml", "--port", "2"] and args.exec_steps is None
    args, rest = SP.parse(["--mode", "warm", "--env-id", "pi05_rc", "--cache-config", "w.yaml", *COMMON, "--", "--non-concurrent",
                           "policy:checkpoint", "--policy.config", "pi05_robocasa"])
    assert rest.count("--non-concurrent") == 1 and rest[0] == "--cache_config=w.yaml"
    assert rest.index("policy:checkpoint") > rest.index("--non-concurrent")  # top-level flags precede the subcommand


@pytest.mark.parametrize("argv", [
    ["--mode", "plain", "--env-id", "pi05_rc"],                                   # no --exec-steps
    ["--mode", "shadow", "--env-id", "pi05_rc"],                                  # no --cache-config
    ["--mode", "full", "--env-id", "pi05_rc", "--cache-config", "c.yaml"],        # cache refused
    ["--mode", "full", "--env-id", "groot_rc"],                                   # wrong family
])
def test_pi05_parse_rejections(argv):
    with pytest.raises(SystemExit):
        SP.parse([*argv, *COMMON])


def test_groot_parse_modes_and_rejections():
    args, rest = SG.parse(["--benchmark", "rc", "--mode", "full", "--env-id", "groot_rc", *COMMON, "--", "--port", "1"])
    assert args.exec_steps == 4 and rest == ["--port", "1"]
    args, rest = SG.parse(["--benchmark", "rc", "--mode", "warm", "--env-id", "groot_rc", "--cache-config", "w.yaml", *COMMON])
    assert rest == ["--cache-config", "w.yaml"]
    args, rest = SG.parse(["--benchmark", "libero", "--mode", "shadow", "--env-id", "groot_libero_10", "--cache-config", "s.yaml", *COMMON])
    assert args.exec_steps is None
    for argv in (["--benchmark", "rc", "--mode", "plain", "--env-id", "groot_rc"],
                 ["--benchmark", "rc", "--mode", "plain", "--env-id", "groot_rc", "--exec-steps", "1", "--cache-config", "c"],
                 ["--benchmark", "rc", "--mode", "shadow", "--env-id", "groot_rc"]):
        with pytest.raises(SystemExit):
            SG.parse([*argv, *COMMON])


def test_env_table_invariants():
    for env in E.ENVS.values():
        assert 0 < env.n_executed <= env.action_dim and env.k_full == env.schedule.num_steps
        for t in env.warm_ts:
            assert 0 < env.remaining_steps(t) < env.k_full
        assert all(0 < k < env.k_full for k in env.k_set)
    assert E.ENVS["pi05_rc"].action_dim == 32 and E.ENVS["pi05_rc"].n_executed == 12
    assert E.ENVS["pi05_libero_spatial"].n_executed == 7 and E.ENVS["pi05_libero_spatial"].action_horizon == 10
    assert E.ENVS["groot_rc"].k_full == 4 and E.ENVS["groot_libero_10"].k_full == 8
