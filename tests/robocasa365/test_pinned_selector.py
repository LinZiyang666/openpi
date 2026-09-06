"""Pure tests for the constraints replayed by the pinned-object selector."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from exp.robocasa365 import select_pinned_objects as selector


def _catalog(monkeypatch, *, groups, categories) -> None:
    """Install the one RoboCasa catalog module imported lazily by the selector."""
    names = (
        "robocasa",
        "robocasa.models",
        "robocasa.models.objects",
        "robocasa.models.objects.kitchen_objects",
    )
    modules = {name: types.ModuleType(name) for name in names}
    for module in modules.values():
        module.__path__ = []
    modules[names[-1]].OBJ_GROUPS = groups
    modules[names[-1]].OBJ_CATEGORIES = categories
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _meta(*, graspable: bool = True, paths=()):
    values = {flag: False for flag in selector.CAPABILITY_FLAGS}
    values["graspable"] = graspable
    return SimpleNamespace(
        **values,
        mjcf_paths=list(paths),
        get_mjcf_kwargs=lambda: {"scale": 1.0},
    )


def test_legal_categories_replays_exclude_capability_and_registry(monkeypatch):
    groups = {
        "all": ["legal", "excluded", "not_graspable", "wrong_registry"],
        "ban": ["excluded"],
    }
    categories = {
        "legal": {"objaverse": _meta()},
        "excluded": {"objaverse": _meta()},
        "not_graspable": {"objaverse": _meta(graspable=False)},
        "wrong_registry": {"lightwheel": _meta()},
    }
    _catalog(monkeypatch, groups=groups, categories=categories)

    assert selector.legal_categories(
        "all",
        "ban",
        {"graspable": True},
        ("objaverse",),
    ) == ["legal"]


def test_target_split_never_selects_from_the_pretrain_slice():
    paths = [f"objects/example/{i}/model.xml" for i in range(10)]
    assert selector.split_slice(paths, "pretrain") == paths[:5]
    assert selector.split_slice(paths, "target") == paths[5:]
    with pytest.raises(ValueError, match="unknown split"):
        selector.split_slice(paths, "validation")


def test_rotate_upright_selects_the_upright_file(monkeypatch, tmp_path):
    # Six instances are needed for the production target split to contain one
    # path (split_th=max(n-5, ceil(n/2))).
    models = [tmp_path / "objects" / "bread" / str(i) / "model.xml" for i in range(6)]
    categories = {"bread": {"objaverse": _meta(paths=[str(path) for path in models])}}
    _catalog(monkeypatch, groups={"all": ["bread"]}, categories=categories)

    choices = selector.candidates_for_slot(
        {"name": "obj", "obj_groups": "all", "rotate_upright": True},
        ("objaverse",),
        "target",
    )

    assert choices == [
        ("bread", "objaverse", str(path).replace("model.xml", "model_upright.xml"))
        for path in models[3:]
    ]


def test_ranked_choices_enforces_max_size(monkeypatch, tmp_path):
    small = tmp_path / "objects" / "small" / "model.xml"
    large = tmp_path / "objects" / "large" / "model.xml"
    small.parent.mkdir(parents=True)
    large.parent.mkdir(parents=True)
    small.write_text("<mujoco/>")
    large.write_text("<mujoco/>")
    categories = {
        "small": {"objaverse": _meta(paths=[str(small)])},
        "large": {"objaverse": _meta(paths=[str(large)])},
    }
    _catalog(monkeypatch, groups={}, categories=categories)
    monkeypatch.setattr(
        selector,
        "candidates_for_slot",
        lambda *_args: [
            ("large", "objaverse", str(large)),
            ("small", "objaverse", str(small)),
        ],
    )
    monkeypatch.setattr(
        selector,
        "object_extent",
        lambda path, _scale: [2.0, 2.0, 2.0] if path == str(large) else [0.5, 0.5, 0.5],
    )

    choices = selector.ranked_choices(
        {"name": "obj", "max_size": [1.0, 1.0, 1.0]},
        ("objaverse",),
        "target",
    )

    assert [choice["category"] for choice in choices] == ["small"]


def test_ranked_choices_fails_when_every_instance_is_too_large(monkeypatch, tmp_path):
    model = tmp_path / "objects" / "large" / "model.xml"
    model.parent.mkdir(parents=True)
    model.write_text("<mujoco/>")
    categories = {"large": {"objaverse": _meta(paths=[str(model)])}}
    _catalog(monkeypatch, groups={}, categories=categories)
    monkeypatch.setattr(
        selector,
        "candidates_for_slot",
        lambda *_args: [("large", "objaverse", str(model))],
    )
    monkeypatch.setattr(selector, "object_extent", lambda *_args: [2.0, 2.0, 2.0])

    with pytest.raises(ValueError, match="no instance survives"):
        selector.ranked_choices(
            {"name": "obj", "max_size": [1.0, 1.0, 1.0]},
            ("objaverse",),
            "target",
        )


def test_choose_for_task_rejects_duplicate_categories(monkeypatch):
    monkeypatch.setattr(
        selector,
        "ranked_choices",
        lambda cfg, *_args: [
            {
                "name": cfg["name"],
                "mjcf_path": f"objects/shared/{cfg['name']}/model.xml",
                "category": "shared",
                "volume": 1.0,
            }
        ],
    )

    with pytest.raises(ValueError, match="unused category"):
        selector.choose_for_task(
            "Task",
            [{"name": "obj"}, {"name": "container"}],
            ("objaverse",),
            "target",
            1,
            1,
            [1_000_000],
            1,
        )


def test_choose_for_task_has_a_bounded_failed_build_search(monkeypatch):
    options = [
        {
            "name": "obj",
            "mjcf_path": f"objects/example/{i}/model.xml",
            "category": f"cat-{i}",
            "volume": float(2 - i),
        }
        for i in range(2)
    ]
    monkeypatch.setattr(selector, "ranked_choices", lambda *_args: list(options))
    monkeypatch.setattr(selector, "task_builds", lambda *_args: "placement failed")

    with pytest.raises(ValueError, match="no pinned combination loaded in 2 attempts"):
        selector.choose_for_task(
            "Task",
            [{"name": "obj"}],
            ("objaverse",),
            "target",
            1,
            1,
            [1_000_000],
            2,
        )
