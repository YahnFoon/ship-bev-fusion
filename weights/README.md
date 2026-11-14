# Checkpoints

The checkpoints are pending a non-commercial research release. If you have a
copy, use the paths below or pass the path to the training or inference command.

| Path | Use |
|---|---|
| `weights/stage1_initializer.pth` | Initialization for the Stage 1 Mask2Former experiments |
| `weights/stage1_reference.pth` | Shared camera segmentation initialization for joint BEV training |
| `weights/ours/seed20260510.pt` | Trained joint model for seed 20260510 |
| `weights/ours/seed20260511.pt` | Trained joint model for seed 20260511 |
| `weights/ours/seed20260512.pt` | Trained joint model for seed 20260512 |

To reproduce Stage 1, start from the in-domain initializer listed above. The
shared joint-training reference was trained with seed `4281387`.

The joint checkpoint contains both the camera segmentation and fusion weights.
Load checkpoints from sources you trust.
