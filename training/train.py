"""Train the proposed ship BEV fusion model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'MVDeTr'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['ours'], default='ours')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage1-config', type=Path, default=ROOT / 'configs/stage1.py')
    parser.add_argument('--stage1-checkpoint', type=Path, default=ROOT / 'weights/stage1_reference.pth')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('MVDETR_MS_DEFORM_ATTN_BACKEND', 'custom')
    os.environ.setdefault('INDOOR_MASK2FORMER_MSDA_BACKEND', 'pytorch')
    import cv2
    import numpy as np
    import torch
    import yaml
    from ship_bev_fusion.config import load_config
    from .epochs import fit

    cfg = load_config(args.config)
    seed = int(cfg['training']['seed'])
    if os.environ.get('PYTHONHASHSEED') != str(seed):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        os.execve(sys.executable, [sys.executable, '-m', 'training.train', *sys.argv[1:]], env)
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA. CPU inference is supported by inference.py.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if cfg['training']['batch_size'] != 6 or cfg['training']['amp']:
        raise ValueError('The released paper training recipe uses batch size 6 and FP32.')
    root = (args.output or Path(cfg['output_dir'])).resolve()
    cfg['output_dir'] = str(root)
    if not args.resume and (root / 'train').exists() and any((root / 'train').iterdir()):
        raise FileExistsError('Output already contains a run; choose another directory or use --resume.')
    (root / 'train').mkdir(parents=True, exist_ok=True)
    p = dict(stage1_config=str(args.stage1_config.resolve()),
             stage1_checkpoint=str(args.stage1_checkpoint.resolve()),
             config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
    (root / 'run_config_resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    best = fit(seed, cfg, p, root, args.resume)
    print('Selected checkpoint:', best)


if __name__ == '__main__':
    main()
