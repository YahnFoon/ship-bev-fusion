# Inference and evaluation

Run from the repository root. Inference requires the
[fusion runtime](INSTALL.md#mask2former-and-joint-fusion), the
[indoor dataset](../data/README.md), and a joint [checkpoint](../weights/README.md).

```bash
python inference.py --config configs/evaluation/ours_water9.yaml \
  --checkpoint weights/ours/seed20260510.pt \
  --split test --output outputs/ours/seed20260510/water9
```

For a single-frame CPU check, add `--device cpu --max-frames 1` and use a new
output directory. For cross-water evaluation, keep the checkpoint and use
`ours_water11.yaml` or `ours_water13.yaml` to set the corresponding calibration
plane.

The evaluator reads masks and scores from the export and calibration from the
dataset. Use `--method` to pass the inference output directory. The prediction
manifest records NPZ paths relative to that directory.

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
COCO AP. Both pair predictions and reference masks by condition and frame index.

The `run_config_resolved.json` files record paths from the machine that ran
inference or evaluation. Replace private paths before sharing these files. Use
`--data-root` to point evaluation at the recipient's dataset copy.
