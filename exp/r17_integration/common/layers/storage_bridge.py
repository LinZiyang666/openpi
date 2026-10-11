"""P6 LibraryView -> shared array-only fitting, without duplicate layer math."""
from ..contracts import ActionStateLayout
from .recipe import LibraryData, fit_library


def fit_library_view(view, *, config, corrector=False):
    """Pass as P6 search.fit.fit_library(..., fitter=fit_library_view).

    config requires layout and ordered_tasks; optional settings are the
    LayoutRecipe constructor knobs. Only P6's allowlisted feature/state/action
    arrays and identity rows are accessed. Outcomes are never consulted.
    """
    import numpy as np
    layout = config["layout"]
    if isinstance(layout, dict):
        layout = dict(layout)
        for key in ("valid_state_indices", "motion_channels", "valid_action_indices", "raw_state_fields"):
            if key in layout:
                layout[key] = tuple(layout[key])
        layout = ActionStateLayout(**layout)
    if not isinstance(layout, ActionStateLayout):
        raise ValueError("fit view requires declared ActionStateLayout")
    task_names = tuple(config["ordered_tasks"])
    if len(set(task_names)) != len(task_names):
        raise ValueError("fit view requires unique canonical ordered tasks")
    tasks = {name: i for i, name in enumerate(task_names)}
    episodes = {ep: i for i, ep in enumerate(dict.fromkeys(view.episode_ids))}
    features = {name.removeprefix("key_"): view.array(name) for name in view.fields if name.startswith("key_")}
    lib = LibraryData(features, view.array("state"), view.array("normalized_actions"),
                      np.asarray([tasks[row["task_name"]] for row in view.rows]),
                      np.asarray([episodes[ep] for ep in view.episode_ids]),
                      np.asarray([row["decision_seq"] for row in view.rows]))
    settings = {k: v for k, v in config.items() if k not in {"layout", "ordered_tasks"}}
    return fit_library(lib, layout=layout, task_count=len(task_names), corrector=corrector, **settings)
