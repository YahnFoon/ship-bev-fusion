# Installation

Run commands from the repository root on Linux or WSL. The paper training
environment used Python 3.8, PyTorch 2.4.0, Torchvision 0.19.0, CUDA 12.1,
MMCV 2.1.0, MMDetection 3.3.0, and MMEngine 0.10.7.

```bash
conda create -n ship-bev python=3.8 -y
conda activate ship-bev
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Dataset preparation, RGB caching and saved-prediction evaluation use the
dependencies above. Run `python -m unittest evaluation.test_statistics`
to check the statistical routines.

## Mask2Former and joint fusion

For GPU runs, build MMCV with the CUDA toolkit matching the PyTorch build.
Then install MMDetection.

```bash
python -m pip install ninja psutil packaging wheel
MMCV_WITH_OPS=1 python -m pip install --no-build-isolation \
  --no-binary mmcv mmcv==2.1.0
python -m pip install mmdet==3.3.0
```

The joint model imports the MVDeTr implementation under
`MVDeTr/multiview_detector/`. The paper configuration uses its ResNet-18
backbone, projection helper, reference map, deformable world-feature
transformer and output head. Our local runtime includes changes to the
reference-point setup and batched fusion. The upstream checkout alone does
not reproduce those changes. The additional runtime files are listed in
[licensing/blocked_source_paths.txt](../licensing/blocked_source_paths.txt).
See the [upstream licensing request](https://github.com/hou-yz/MVDeTr/issues/10)
and [checkpoint paths](../weights/README.md).

For CUDA fusion with the local runtime available, build the included attention
operator with the same toolkit and interpreter.

```bash
python -m pip install --no-build-isolation ./MVDeTr/multiview_detector/models/ops
```

Training selects the custom fusion kernel and the PyTorch implementation of
Mask2Former attention. `inference.py --device cpu` also uses PyTorch for fusion
attention, so it can run without the custom fusion extension. CPU inference
still imports MMCV and MMDetection.
