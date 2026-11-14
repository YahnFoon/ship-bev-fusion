# Training

Run from the repository root with the [dataset](../data/README.md) under
`data/indoor`. Train and validate on the fixed condition splits at 81.5 mm.
The three runs use seeds `20260510`, `20260511`, and `20260512`.

## Stage 1 camera segmentation

Prepare the four-view COCO image dataset.

```bash
python -m scripts.build_indoor_20260510_aligned_mmdet_dataset \
  --data-root data/indoor \
  --video-dir Video/waterLevel_9/waterlevel_9_synced_conditions \
  --mask-dir Video/waterLevel_9/multiview_dino_sam2_masks_zero_valid \
  --split-manifest splits/water9.json \
  --camera-ids 221 222 223 224 \
  --splits train val \
  --output-root data/stage1

python -m training.stage1 --seed 20260510 --output outputs/stage1/seed20260510
```

The builder produces 53,736 training images and 6,968 validation images.
Mask2Former-R50 trains for 12 epochs at batch size 16 with AdamW
and a learning rate of 2e-6. The runner selects by validation segmentation AP.
The experiment starts from `weights/stage1_initializer.pth`,
which was fine-tuned on earlier data outside this dataset release.

## Stage 2 joint BEV segmentation

With the [fusion runtime](INSTALL.md#mask2former-and-joint-fusion) and the shared
`weights/stage1_reference.pth` available, build the RGB cache and train.

```bash
PYTHONPATH=MVDeTr python -m ship_bev_fusion.mvdetr_rgb_precompute_cache \
  --config configs/ours/seed20260510.yaml

python -m training.train --config configs/ours/seed20260510.yaml
```

Reuse the cache for the other two seeds. Training uses 24 epochs, batch size 6,
FP32 with TF32 disabled, and AdamW with learning rates of 1e-4 for Mask2Former
and 2e-4 for the fusion model. The RGB overlay uses a background weight of 0.25.
The BEV loss uses positive BCE weight 10 and Dice weight 1. The reported runs
used an 80 GB GPU.

The trainer saves the best epoch by validation Dice and the latest epoch for
resuming. Use `--resume` with the same configuration and output directory. Both
checkpoints contain the resolved local configuration. The checkpoints listed
in the [weights directory](../weights/README.md) are prepared separately for
sharing.
