"""Fine-tune the independent Mask2Former front end."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/stage1.py')
    p.add_argument('--seed', type=int, default=20260510)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    cfg.randomness = dict(seed=args.seed, deterministic=False, diff_rank_seed=False)
    cfg.work_dir = str(args.output.resolve())
    cfg.launcher = 'none'
    cfg.resume = args.resume
    if args.resume:
        cfg.load_from = None
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use an empty output directory or --resume.')
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
