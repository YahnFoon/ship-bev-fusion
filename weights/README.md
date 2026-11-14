# Checkpoints

The checkpoint files are pending a separate non-commercial research release.
If you already have the local checkpoints, place them at these paths or pass
their paths to the corresponding command.

| Path | Use |
|---|---|
| `weights/stage1_initializer.pth` | Initialization for the Stage 1 Mask2Former experiments |
| `weights/stage1_reference.pth` | Shared camera segmentation initialization for joint BEV training |
| `weights/ours/seed20260510.pt` | Trained joint model for seed 20260510 |
| `weights/ours/seed20260511.pt` | Trained joint model for seed 20260511 |
| `weights/ours/seed20260512.pt` | Trained joint model for seed 20260512 |

For an exact Stage 1 rerun, use the same in-domain initializer as the paper.
The shared joint-training reference uses seed `4281387`.

The joint checkpoint contains both the camera segmentation and fusion weights.
Load checkpoints from sources you trust.
