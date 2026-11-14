# Ship BEV Fusion

Ship BEV Fusion segments ships in bird's-eye view (BEV) from four synchronized
cameras. Mask2Former estimates a ship mask for each view. The fusion model
weights the RGB inputs with these masks, projects their features onto the water
plane, and predicts the BEV mask with deformable attention.

Use the [installation instructions](docs/INSTALL.md) to set up the environment
for dataset preparation and evaluation of saved predictions. The joint model
requires the MVDeTr runtime files and checkpoints listed there. Those files are
pending separate release.

## Data

The indoor dataset contains synchronized videos, COCO RLE masks, camera
calibration, and fixed splits for three water heights. Download the archives or
point the configs to a local copy using the [dataset instructions](data/README.md).

| Water height | Split | Conditions | Retained frames |
|---|---|---:|---:|
| 81.5 mm (`water9`) | Train | 76 | 13,434 |
| 81.5 mm (`water9`) | Validation | 9 | 1,742 |
| 81.5 mm (`water9`) | Test | 9 | 1,530 |
| 101.5 mm (`water11`) | Test | 93 | 20,749 |
| 121.5 mm (`water13`) | Test | 92 | 20,292 |

## Running the code

- [Installation](docs/INSTALL.md)
- [Training](docs/TRAINING.md)
- [Inference and evaluation](docs/EVALUATION.md)
- [Checkpoint paths](weights/README.md)

The fusion model builds on [MVDeTr](https://github.com/hou-yz/MVDeTr) and uses
operators from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
The image segmentation front end uses [MMDetection](https://github.com/open-mmlab/mmdetection).
See [Licensing](LICENSING.md) for the original code, third-party code and dataset terms.
