"""Unit tests for ``exp/common/_unit_key.py`` (cleanup/03a, plan P0-5).

Two kinds of coverage:

1. Round-trip per class (decode(encode(x)) == x, invalid inputs raise).
2. **Golden byte-equal** against the pre-cleanup string format for each
   runner — hand-built fixed inputs, not round-trip. Pre-cleanup
   state-file key strings must survive cleanup/03b unchanged (plan §7
   D2 + G2 Watchpoint #2).
"""

from __future__ import annotations

import pytest

from exp.common._unit_key import DeviateKey, SpawnKey, Step1bKey


# --- Step1bKey ----------------------------------------------------------


class TestStep1bKey:
    def test_encode_golden(self):
        assert Step1bKey(task_id=3, init_idx=17).encode() == "3:17"
        assert Step1bKey(task_id=0, init_idx=0).encode() == "0:0"

    def test_decode_golden(self):
        assert Step1bKey.decode("3:17") == Step1bKey(task_id=3, init_idx=17)

    def test_roundtrip(self):
        k = Step1bKey(task_id=9, init_idx=42)
        assert Step1bKey.decode(k.encode()) == k

    def test_decode_rejects_missing_field(self):
        with pytest.raises(ValueError, match="Step1bKey"):
            Step1bKey.decode("3")

    def test_decode_rejects_extra_field(self):
        with pytest.raises(ValueError, match="Step1bKey"):
            Step1bKey.decode("3:17:99")

    def test_decode_rejects_non_int(self):
        with pytest.raises(ValueError, match="Step1bKey"):
            Step1bKey.decode("3:x")


# --- DeviateKey ---------------------------------------------------------


class TestDeviateKey:
    def test_encode_phase1_golden(self):
        # Pre-cleanup format: f"{self.common.config_id}:{ep}:{s}"
        k = DeviateKey(cfg="clip_w7_d4", ep="task_0/episode_1", sample_idx=5)
        assert k.encode() == "clip_w7_d4:task_0/episode_1:5"

    def test_encode_phase2_golden(self):
        # Pre-cleanup format: f"{self.common.config_id}:{ep}"
        k = DeviateKey(cfg="clip_w7_d4", ep="task_0/episode_1", sample_idx=None)
        assert k.encode() == "clip_w7_d4:task_0/episode_1"

    def test_encode_phase1_sample_idx_zero(self):
        """``sample_idx=0`` must encode the Phase1 format, not Phase2.
        ``None`` (Phase2) and ``0`` (Phase1 first sample) must be distinct.
        """
        zero = DeviateKey(cfg="c", ep="task_0/episode_0", sample_idx=0)
        none = DeviateKey(cfg="c", ep="task_0/episode_0", sample_idx=None)
        assert zero.encode() == "c:task_0/episode_0:0"
        assert none.encode() == "c:task_0/episode_0"
        assert zero != none

    def test_decode_phase1_golden(self):
        assert DeviateKey.decode("clip_w7_d4:task_0/episode_1:5") == DeviateKey(
            cfg="clip_w7_d4", ep="task_0/episode_1", sample_idx=5
        )

    def test_decode_phase2_golden(self):
        assert DeviateKey.decode("clip_w7_d4:task_0/episode_1") == DeviateKey(
            cfg="clip_w7_d4", ep="task_0/episode_1", sample_idx=None
        )

    def test_roundtrip_phase1(self):
        k = DeviateKey(cfg="clip_w7_d4", ep="task_3/episode_7", sample_idx=0)
        assert DeviateKey.decode(k.encode()) == k

    def test_roundtrip_phase2(self):
        k = DeviateKey(cfg="clip_w7_d4", ep="task_3/episode_7", sample_idx=None)
        assert DeviateKey.decode(k.encode()) == k

    def test_decode_rejects_empty(self):
        with pytest.raises(ValueError, match="DeviateKey"):
            DeviateKey.decode(":task_0/episode_0")
        with pytest.raises(ValueError, match="DeviateKey"):
            DeviateKey.decode("cfg:")

    def test_decode_rejects_too_many_segments(self):
        with pytest.raises(ValueError, match="DeviateKey"):
            DeviateKey.decode("cfg:task_0/episode_0:1:extra")


# --- SpawnKey -----------------------------------------------------------


class TestSpawnKey:
    def test_encode_golden(self):
        # Pre-cleanup format: f"{cfg}:{ep}:s{s}:n{n}:k{k_idx}"
        k = SpawnKey(cfg="clip_w7_d4", ep="task_0/episode_0", s=12, n=1, k_idx=0)
        assert k.encode() == "clip_w7_d4:task_0/episode_0:s12:n1:k0"

    def test_encode_multi_digit_golden(self):
        k = SpawnKey(cfg="clip_w7_d4", ep="task_1/episode_0", s=18, n=3, k_idx=2)
        assert k.encode() == "clip_w7_d4:task_1/episode_0:s18:n3:k2"

    def test_decode_golden(self):
        k = SpawnKey.decode("clip_w7_d4:task_0/episode_0:s12:n1:k0")
        assert k == SpawnKey(
            cfg="clip_w7_d4", ep="task_0/episode_0", s=12, n=1, k_idx=0
        )

    def test_roundtrip(self):
        k = SpawnKey(cfg="baseline_random", ep="task_7/episode_3", s=0, n=20, k_idx=4)
        assert SpawnKey.decode(k.encode()) == k

    def test_decode_rejects_missing_prefix(self):
        with pytest.raises(ValueError, match="SpawnKey"):
            SpawnKey.decode("cfg:task_0/episode_0:12:1:0")  # no s/n/k prefixes

    def test_decode_rejects_malformed(self):
        with pytest.raises(ValueError, match="SpawnKey"):
            SpawnKey.decode("cfg:task_0/episode_0:s12:n1")  # missing k segment

    def test_decode_rejects_negative(self):
        # Regex is \d+, so "-1" fails to match (ValueError, not int-cast error).
        with pytest.raises(ValueError, match="SpawnKey"):
            SpawnKey.decode("cfg:task_0/episode_0:s-1:n1:k0")


# --- Cross-schema: encoded strings do not collide ----------------------


def test_schemas_do_not_collide_on_concrete_samples():
    """Concrete pre-cleanup samples from the three runners must produce
    disjoint strings — no accidental ``decode`` cross-parsing."""
    step1b = Step1bKey(task_id=3, init_idx=17).encode()
    deviate_p1 = DeviateKey(cfg="c", ep="task_0/episode_0", sample_idx=5).encode()
    deviate_p2 = DeviateKey(cfg="c", ep="task_0/episode_0", sample_idx=None).encode()
    spawn = SpawnKey(cfg="c", ep="task_0/episode_0", s=1, n=2, k_idx=0).encode()

    assert len({step1b, deviate_p1, deviate_p2, spawn}) == 4

    # Step1bKey refuses to decode a DeviateKey / SpawnKey (different arity).
    for foreign in (deviate_p1, deviate_p2, spawn):
        with pytest.raises(ValueError):
            Step1bKey.decode(foreign)

    # SpawnKey refuses anything lacking s/n/k prefixes.
    for foreign in (step1b, deviate_p1, deviate_p2):
        with pytest.raises(ValueError):
            SpawnKey.decode(foreign)
