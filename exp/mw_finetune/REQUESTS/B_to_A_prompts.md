# B → A: source/harness prompt difference

B preserves the source task_index→text mapping exactly and passes canonical
obs.prompt unchanged in serving. Source task_id and task_index have different
orders from A's table; neither numeric ID is used by B inference.

Set comparison: every harness prompt exists in the training source. The sole
extra source text is `Push the puck back toward the robot to the goal`, the
LeRobot rewrite for push-back. A deliberately keeps RLinf's `Push the puck to
a goal` for push-back-v3, as documented in exp/metaworld/tasks.py. B does not
rewrite either the data or serving input. Please disclose this one-task prompt
shift in evaluation reports, or coordinate a separate dataset-prompt sensitivity
run if useful. B's recipe/handback record the difference.
