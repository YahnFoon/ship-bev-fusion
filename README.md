# Ship BEV Fusion

Ship BEV Fusion segments ships in bird's-eye view (BEV) from four synchronized
cameras. Mask2Former estimates a ship mask for each view. The fusion model
weights the RGB inputs with these masks, projects their features onto the water
plane, and predicts the BEV mask with deformable attention.

Use the [installation instructions](docs/INSTALL.md) to prepare the dataset
or evaluate saved predictions. Joint training and inference also require the
adapted MVDeTr source files and Ship BEV Fusion checkpoints. The source files
are awaiting redistribution permission, and the checkpoints are pending a
separate release.

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

## Acknowledgments

We thank Yunzhong Hou and Liang Zheng for sharing
[MVDeTr](https://github.com/hou-yz/MVDeTr). Our BEV fusion stage builds on their
work, including the deformable fusion module, reference-point generation,
camera projection helpers, and ResNet-18 backbone. See their
[ACM Multimedia 2021 paper](https://doi.org/10.1145/3474085.3475310).

The fusion stage uses attention operators from
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
The image segmentation front end uses [MMDetection](https://github.com/open-mmlab/mmdetection).

See [Licensing](LICENSING.md) for code and dataset terms.
