import random
from pathlib import Path
import numpy as np
import torch

def build(cfg, paths):
    from scripts import train_indoor_20260510_mask2former_deformtrans_stage2 as stage2

    use_cache = bool(cfg['training'].get('use_cache', False))
    train = stage2.Stage2AlignedDataset('train', cfg, bev_only=True, use_rgb_cache=use_cache)
    val = stage2.Stage2AlignedDataset('val', cfg, bev_only=True, use_rgb_cache=use_cache)
    mask = stage2.build_detector(Path(paths['stage1_config']), Path(paths['stage1_checkpoint'])).cuda()
    fusion = stage2.build_stage2_fusion_model(train, cfg, 'rgb_mask_overlay').cuda()
    optimizer = torch.optim.AdamW(
        [
            dict(params=mask.parameters(), lr=1e-4),
            dict(params=fusion.parameters(), lr=2e-4),
        ],
        weight_decay=1e-4,
    )
    assert len(train) == 13434 and len(val) == 1742
    return stage2, train, val, (mask, fusion), optimizer, stage2.collate_stage2


def worker_init(_):
    import cv2
    cv2.setNumThreads(0)
    random.seed(torch.initial_seed()%2**32)
    np.random.seed(torch.initial_seed()%2**32)
