import argparse
import hashlib
import json
import math
import torch
from torch.utils.data import DataLoader, RandomSampler
from .runtime import build, worker_init
from .recovery import capture_rng, restore_rng, save_epoch, repair_best, write_history

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def epoch_args():
    return argparse.Namespace(freeze_mask2former=False,grad_accumulation_steps=1,gate_monitor_interval=0,
        bev_only=True,mask2former_loss_weight=0.,stage2_input_mode='rgb_mask_overlay',roi_background_weight=0.25,
        pos_weight=10.,dice_weight=1.,boundary_iou_weight=0.,boundary_kernel_size=5,grad_clip=1.,log_interval=100)

def loaders(train,val,collate,seed,workers,batch=6):
    generators={k:torch.Generator().manual_seed(seed+offset) for k,offset in [('shuffle',0),('train_workers',1),('val_workers',2)]}
    common=dict(batch_size=batch,num_workers=workers,pin_memory=True,collate_fn=collate,
                worker_init_fn=worker_init,persistent_workers=workers>0)
    train_loader=DataLoader(train,sampler=RandomSampler(train,generator=generators['shuffle']),
                            generator=generators['train_workers'],**common)
    val_loader=DataLoader(val,shuffle=False,generator=generators['val_workers'],**common)
    return train_loader,val_loader,generators

def fit(seed, cfg, p, root, resume):
    method = 'ours'
    train_dir = root / 'train'
    last = train_dir / 'stage2_last_checkpoint.pt'
    best = train_dir / 'stage2_best_checkpoint.pt'
    history_path = train_dir / 'stage2_training_history.csv'
    stage2, train, val, (mask, fusion), opt, collate = build(cfg, p)
    device = torch.device('cuda')
    coverage=[]
    for dataset in (train,val):
        records=[(str(r.condition),int(r.frame_index)) for r in dataset.records]
        assert len(set(records))==len(records)
        coverage.append(dict(samples=len(records),conditions=len({c for c,_ in records})))
    assert sorted(c['samples'] for c in coverage)==[1742,13434]
    (root/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    train_loader,val_loader,generators=loaders(train,val,collate,seed,cfg['training']['num_workers'])
    history=[];best_epoch=0;best_dice=-math.inf;start=1
    if resume:
        checkpoint=torch.load(last,map_location='cpu',weights_only=False)
        assert checkpoint['recovery_version']==2
        assert checkpoint['protocol_sha256']==p['config_sha256']
        assert checkpoint['method']==method and checkpoint['seed']==seed
        assert checkpoint['config']==cfg, 'Do not change configuration during recovery'
        mask.load_state_dict(checkpoint['mask2former'], strict=True)
        fusion.load_state_dict(checkpoint['fusion_model'], strict=True)
        opt.load_state_dict(checkpoint['optimizer'])
        history=checkpoint['history'];start=checkpoint['epoch']+1
        best_epoch=checkpoint['best_epoch'];best_dice=checkpoint['best_dice']
        repair_best(last,best,checkpoint);write_history(history_path,history)
        restore_rng(checkpoint['rng'],generators);del checkpoint
        print('RESUME: complete epoch %d -> %d; model, optimizer, RNG, sampler and history restored'%(start-1,start),flush=True)
    for epoch in range(start, cfg['training']['epochs'] + 1):
        loss = stage2.train_one_epoch(
            mask_model=mask, fusion_model=fusion, dataset=train, loader=train_loader,
            optimizer=opt, device=device, args=epoch_args(), epoch=epoch,
        )
        metrics = stage2.evaluate(
            mask, fusion, val, val_loader, device,
            stage2_input_mode='rgb_mask_overlay', roi_background_weight=0.25,
        )
        assert math.isfinite(loss) and all(math.isfinite(float(v)) for v in metrics.values())
        history.append(dict(epoch=epoch,train_loss=loss,**{'val_'+k:float(v) for k,v in metrics.items()}))
        if metrics['dice']>best_dice:best_dice=float(metrics['dice']);best_epoch=epoch
        checkpoint=dict(epoch=epoch,optimizer=opt.state_dict(),config=cfg,fusion_config=cfg,val_metrics=metrics,
            history=history,best_epoch=best_epoch,best_dice=best_dice,rng=capture_rng(generators),recovery_version=2,
            protocol_sha256=p['config_sha256'],method=method, seed=seed, mask2former=mask.state_dict(), fusion_model=fusion.state_dict())
        save_epoch(last,best,checkpoint);write_history(history_path,history);del checkpoint
        status=dict(status='training',method=method,seed=seed,epoch=epoch,train_loss=loss,val_metrics=metrics,
                    best_epoch=best_epoch,best_val_dice=best_dice)
        (root/'progress.json').write_text(json.dumps(status)+'\n');print(json.dumps(status),flush=True)
    assert [int(r['epoch']) for r in history]==list(range(1, cfg['training']['epochs'] + 1))
    steps=sorted({int(s['step']) for s in opt.state.values() if 'step' in s});assert steps == [math.ceil(len(train) / 6) * cfg['training']['epochs']], steps
    expected=max(history,key=lambda r:r['val_dice']);assert expected['epoch']==best_epoch
    checkpoint=torch.load(best,map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==best_epoch and math.isclose(checkpoint['val_metrics']['dice'],best_dice,abs_tol=1e-8)
    audit=dict(status='passed',best_epoch=best_epoch,best_val_dice=best_dice,optimizer_steps=steps,
               checkpoint_sha256=sha(best),recovery_version=2)
    (root/'training_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return best
