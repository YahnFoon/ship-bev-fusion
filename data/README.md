# Indoor Dataset v1.0

Synchronized multi-view ship videos, COCO RLE masks, camera calibration and
fixed experiment splits for bird’s-eye-view (BEV) ship segmentation.

| Water height | Split | Conditions | Retained synchronized frames |
|---|---|---:|---:|
| 81.5 mm (`water9`) | Train | 76 | 13,434 |
| 81.5 mm (`water9`) | Validation | 9 | 1,742 |
| 81.5 mm (`water9`) | Test | 9 | 1,530 |
| 101.5 mm (`water11`) | Test | 93 | 20,749 |
| 121.5 mm (`water13`) | Test | 92 | 20,292 |

The four input cameras are `221`, `222`, `223` and `224`. The `GT` camera
provides reference labels. The videos contain the original synchronized clips.
The split manifests specify each condition and the retained frame indices.
Frames with a COCO image record and no annotation represent empty masks.

Use your existing copy if you already have the dataset. For inference and
training, set `dataset.root` in the YAML config to the path of that copy’s
`data/indoor` directory. For dataset preparation and saved-prediction
evaluation, pass that directory as `--data-root`.

Otherwise, download the files from the [dataset release](https://github.com/YahnFoon/ship-bev-fusion/releases/tag/dataset-v1.0).
Start with `metadata.zip`, which includes calibration, the file manifest, the
license notice and all three fixed splits under `data/indoor/splits/`.
Then download the ZIP files for the water heights and splits you need.
`ASSETS.json` lists the conditions and retained frame counts in each ZIP.
Extract all downloaded ZIPs into the same destination. Each archive preserves
its relative paths under `data/`. The repeated `data/LICENSE` file is identical
across the ZIPs.

To verify the archives on Linux or WSL, place `SHA256SUMS` beside the downloaded
ZIPs and run `sha256sum -c --ignore-missing SHA256SUMS`. After extracting the
whole dataset, run `sha256sum -c data/SHA256SUMS` from the destination directory.
Run `python tools/validate_data.py` from the code repository root to check frame labels and splits.

The indoor dataset is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Credit the Ship BEV Fusion contributors and link to
[Ship BEV Fusion Indoor Dataset](https://github.com/YahnFoon/ship-bev-fusion/releases/tag/dataset-v1.0).
Please retain the license link and indicate changes if you adapt the dataset.
