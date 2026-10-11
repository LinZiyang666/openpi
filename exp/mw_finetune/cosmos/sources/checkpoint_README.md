---
license: other
library_name: cosmos-policy
pipeline_tag: robotics
---

# Cosmos Policy MetaWorld Predict2 2B

FlowDAgger OSS release artifact for the Cosmos 2B MetaWorld backend.

Files:
- `consolidated.pt`: Cosmos Policy Predict2 2B checkpoint finetuned on MetaWorld data for the release recipe.
- `dataset_statistics.json`: action/proprio normalization statistics.
- `t5_embeddings.pkl`: cached text embeddings for the MetaWorld prompts used by the example.

The release recipe targets MetaWorld `assembly-v3` with prompt `Pick up a nut and place it onto a peg`. This checkpoint intentionally leaves headroom: the frozen base policy is about 52% SR over 25 eval episodes in the reference run, and FlowDAgger full-mode steering reaches about 84-88%.

The checkpoint expects MetaWorld gripper proprio encoded as `[g, -g]`; use `--gripper_qpos_mode opposed` with the FlowDAgger Cosmos runner.
