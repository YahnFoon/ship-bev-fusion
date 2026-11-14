"""Atomic epoch checkpoints, including optimizer, RNG, shuffle and history."""
import csv
import io
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch


def atomic_save(payload, path):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_copy(source, target):
    target = Path(target)
    temp = target.with_name(target.name + '.tmp')
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def capture_rng(generators):
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        generators={name: generator.get_state() for name, generator in generators.items()},
    )


def restore_rng(state, generators):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    for name, generator in generators.items():
        generator.set_state(state['generators'][name])


def save_epoch(last, best, payload):
    atomic_save(payload, last)
    if payload['best_epoch'] == payload['epoch']:
        atomic_copy(last, best)


def repair_best(last, best, payload):
    if payload['best_epoch'] == payload['epoch']:
        atomic_copy(last, best)
    else:
        assert Path(best).exists(), 'Previous best checkpoint is missing'
        checkpoint = torch.load(best, map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == payload['best_epoch'], 'Best checkpoint does not match committed history'


def write_history(path, history):
    text = io.StringIO()
    fields = ['epoch', 'train_loss', 'val_iou', 'val_dice', 'val_precision', 'val_recall']
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    writer.writerows(history)

    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(text.getvalue())
    temp.replace(path)
