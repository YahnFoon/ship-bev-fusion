"""Export ship BEV predictions and aligned ground-truth masks."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'MVDeTr'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['ours'], default='ours')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--stage1-config', type=Path, default=ROOT / 'configs/stage1.py')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--water', choices=[9, 11, 13], type=int)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--max-frames', type=int)
    args = p.parse_args()
    if args.device == 'cpu':
        os.environ.setdefault('OPENCV_FFMPEG_THREADS', '1')
    os.environ.setdefault('MVDETR_MS_DEFORM_ATTN_BACKEND', 'pytorch' if args.device == 'cpu' else 'custom')
    os.environ.setdefault('INDOOR_MASK2FORMER_MSDA_BACKEND', 'pytorch')
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from ship_bev_fusion.config import load_config
    from prediction_io import StreamingSplitWriter
    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_config(args.config)
    if args.water:
        w = args.water
        z = {9:81.5, 11:101.5, 13:121.5}[w]
        d = cfg['dataset']
        d.update(video_dir=f'Video/waterLevel_{w}/waterlevel_{w}_synced_conditions',
                 mask_dir=f'Video/waterLevel_{w}/multiview_dino_sam2_masks' + ('_zero_valid' if w == 9 else ''),
                 split_manifest=f'splits/water{w}.json', plane_z_mm=z)
        d['target_mask_dir'] = d['mask_dir']
        cfg['model']['z_mm'] = z
    cfg['dataset'].update(use_cache=False, cache_dir=None)
    cfg.setdefault('training', {})['use_cache'] = False
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    device = torch.device(args.device)
    from scripts.train_indoor_20260510_mask2former_deformtrans_stage2 import Stage2AlignedDataset, collate_stage2
    from scripts.evaluate_indoor_20260510_mask2former_deformtrans_stage2 import build_models, forward_stage2

    ds = Stage2AlignedDataset(args.split, cfg, bev_only=True, use_rgb_cache=False)
    state['fusion_config'] = cfg
    mask, fusion = build_models(state, ds, argparse.Namespace(mmdet_config=args.stage1_config), device)
    mask.eval()
    fusion.eval()
    del state
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory.')
    out.mkdir(parents=True, exist_ok=True)
    n = len(ds) if args.max_frames is None else min(len(ds), args.max_frames)
    if n < 1:
        raise ValueError('At least one frame is required.')
    selected = Subset(ds, range(n)) if n < len(ds) else ds
    loader = DataLoader(selected, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_stage2)
    (out / 'run_config_resolved.json').write_text(json.dumps(cfg, indent=2)+'\n')
    writer = StreamingSplitWriter(split=args.split, output_dir=out, threshold=.5,
                                  run_name='paper', mode='stage2')
    with torch.inference_mode():
        for batch in loader:
            prediction = forward_stage2(
                mask_model=mask, fusion_model=fusion, dataset=ds, batch=batch, device=device,
            )
            logits = prediction['logits'].detach().float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise ValueError('Non-finite BEV prediction.')
            for i, condition in enumerate(batch['condition']):
                writer.append(condition=str(condition), frame_index=int(batch['frame_index'][i]),
                              logits_hw=logits[i,0], target_hw=batch['target'][i,0].numpy())
    rows, _ = writer.finalize()
    if len(rows) != n:
        raise RuntimeError('Exported frame count differs from the requested dataset.')
    (out / 'export_complete.json').write_text(json.dumps(dict(method=args.method, frames=n,
        full_split=n == len(ds), split=args.split, threshold=.5, device=args.device), indent=2)+'\n')
    print('Exported', n, 'frames to', out)


if __name__ == '__main__':
    main()
