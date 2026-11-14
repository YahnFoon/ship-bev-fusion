# Inference and evaluation

Run from the repository root. The complete model requires the
[fusion runtime](INSTALL.md#mask2former-and-joint-fusion), an
[indoor dataset](../data/README.md) copy and a joint
[checkpoint](../weights/README.md).

```bash
python inference.py --config configs/evaluation/ours_water9.yaml \
  --checkpoint weights/ours/seed20260510.pt \
  --split test --output outputs/ours/seed20260510/water9
```

For a single-frame CPU check, add `--device cpu --max-frames 1` and use a new
output directory. The output path is also needed by the evaluation commands. Use the same checkpoint and the corresponding
`ours_water11.yaml` or `ours_water13.yaml` for cross-water evaluation.
Each configuration sets the calibration plane to that test water height.

Evaluate exported predictions with the same water height and split. The
evaluator uses masks and scores from the export and calibration from the
dataset. It does not load the fusion model.

```bash
python -m scripts.evaluate_indoor_20260510_saved_predictions \
  --method ours=outputs/ours/seed20260510/water9 \
  --data-root data/indoor --splits test \
  --output-dir outputs/ours/seed20260510/water9_metrics

python -m evaluation.condition_ap \
  --prediction-root outputs/ours/seed20260510/water9 \
  --method ours --metrics-dir outputs/ours/seed20260510/water9_metrics \
  --output outputs/ours/seed20260510/water9_condition_ap
```

The evaluator reports overlap, contour, centroid, clearance, heading and
COCO AP metrics. The condition AP command reports per-condition and pooled
COCO AP. Both use the exported sample identities to pair predictions and
reference masks.
