# Ship BEV Fusion

This project segments ships in a bird's-eye-view (BEV) grid from four synchronized
cameras. Mask2Former estimates a foreground mask in each view. The fusion model
uses the masks to weight RGB inputs, projects features onto the water plane, and
predicts a BEV mask with deformable attention.

The proposed method's training, inference and evaluation code is here. Running
the joint model also requires the MVDeTr components and checkpoints listed in
[Installation](docs/INSTALL.md). Those files are pending separate release.
Dataset preparation and evaluation of saved predictions can run without them.

## Data

The indoor dataset contains synchronized videos, COCO RLE masks, calibration and
fixed condition splits for three water heights. Use an existing copy under
`data/indoor` or follow the [dataset instructions](data/README.md).

| Water height | Split | Conditions | Retained frames |
|---|---|---:|---:|
| 81.5 mm (`water9`) | Train | 76 | 13,434 |
| 81.5 mm (`water9`) | Validation | 9 | 1,742 |
| 81.5 mm (`water9`) | Test | 9 | 1,530 |
| 101.5 mm (`water11`) | Test | 93 | 20,749 |
| 121.5 mm (`water13`) | Test | 92 | 20,292 |

## Experiments

- [Installation](docs/INSTALL.md)
- [Training](docs/TRAINING.md)
- [Inference and evaluation](docs/EVALUATION.md)
- [Checkpoint paths](weights/README.md)

The fusion model builds on [MVDeTr](https://github.com/hou-yz/MVDeTr) and uses
operators from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
The image segmentation front end uses [MMDetection](https://github.com/open-mmlab/mmdetection).
See [Licensing](LICENSING.md) for the original code, third-party code and dataset terms.
