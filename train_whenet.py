"""
WHENet Training Script (PyTorch)
=================================
Train EfficientNet-B0 with bin-classification + residual loss on 300W-LP,
evaluate on AFLW2000.

Usage:
    python train_whenet.py \
        --data data \
        --epochs 100 \
        --batch-size 128 \
        --lr 1e-4 \
        --save-path weights

Key design choices (matching WHENet paper):
  - EfficientNet-B0 pretrained ImageNet
  - Yaw:   120 bins × 3° = ±180° full range
  - Pitch: 66  bins × 3° = ±99°
  - Roll:  66  bins × 3° = ±99°
  - Loss:  alpha*CE + beta*MSE  (default alpha=1, beta=2)
  - Optimizer: Adam, lr=1e-4
  - LR scheduler: MultiStepLR
"""

import os
import sys
import time
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from models.whenet import WHENet
from utils.whenet_loss import WHENetLoss
from utils.whenet_dataset import (
    WHENetPose300W,
    WHENetAFLW2000,
    TRAIN_TRANSFORM,
    EVAL_TRANSFORM,
)

# ─────────────────────────────────────────────────────────────────────────────
# logger
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
LOGGER = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='WHENet training')
    p.add_argument('--data',        type=str,   default='data',    help='Root dir containing 300W_LP/ and AFLW2000/')
    p.add_argument('--epochs',      type=int,   default=100,       help='Total training epochs')
    p.add_argument('--batch-size',  type=int,   default=128,       help='Batch size')
    p.add_argument('--lr',          type=float, default=1e-4,      help='Initial learning rate')
    p.add_argument('--alpha',       type=float, default=1.0,       help='CE loss weight')
    p.add_argument('--beta',        type=float, default=2.0,       help='MSE loss weight')
    p.add_argument('--num-workers', type=int,   default=8,         help='DataLoader workers')
    p.add_argument('--milestones',  type=int,   nargs='+',         default=[40, 70], help='LR decay epochs')
    p.add_argument('--gamma',       type=float, default=0.1,       help='LR decay factor')
    p.add_argument('--save-path',   type=str,   default='weights', help='Checkpoint save dir')
    p.add_argument('--checkpoint',  type=str,   default=None,      help='Resume from checkpoint')
    p.add_argument('--print-freq',  type=int,   default=100,       help='Log every N batches')
    p.add_argument('--no-pretrain', action='store_true',           help='Disable ImageNet pretrain')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def angle_mae(pred_deg: torch.Tensor, gt_deg: torch.Tensor) -> torch.Tensor:
    """MAE with wraparound correction (handles ±180° boundary)."""
    diff = torch.abs(pred_deg - gt_deg)
    diff = torch.min(diff, 360.0 - diff)
    return diff


# ─────────────────────────────────────────────────────────────────────────────
# train / eval
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, optimizer, loader, device, epoch, params):
    model.train()
    loss_meter   = AverageMeter()
    ce_meter     = AverageMeter()
    mse_meter    = AverageMeter()
    t0 = time.time()

    for i, (imgs, yaw_gt, pitch_gt, roll_gt) in enumerate(loader):
        imgs    = imgs.to(device, non_blocking=True)
        yaw_gt  = yaw_gt.to(device,   non_blocking=True)
        pitch_gt= pitch_gt.to(device, non_blocking=True)
        roll_gt = roll_gt.to(device,  non_blocking=True)

        yaw_l, pitch_l, roll_l = model(imgs)
        loss, ce_loss, mse_loss = criterion(
            yaw_l, pitch_l, roll_l,
            yaw_gt, pitch_gt, roll_gt
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        loss_meter.update(loss.item(), bs)
        ce_meter.update(ce_loss.item(), bs)
        mse_meter.update(mse_loss.item(), bs)

        if (i + 1) % params.print_freq == 0:
            elapsed = time.time() - t0
            LOGGER.info(
                f'Epoch [{epoch+1}/{params.epochs}] '
                f'Step [{i+1}/{len(loader)}] '
                f'Loss: {loss_meter.avg:.4f}  '
                f'CE: {ce_meter.avg:.4f}  '
                f'MSE: {mse_meter.avg:.4f}  '
                f'Time: {elapsed:.1f}s'
            )
            t0 = time.time()

    LOGGER.info(
        f'Epoch [{epoch+1}/{params.epochs}] Summary  '
        f'Loss: {loss_meter.avg:.4f}  '
        f'CE: {ce_meter.avg:.4f}  '
        f'MSE: {mse_meter.avg:.4f}'
    )
    return loss_meter.avg


@torch.no_grad()
def evaluate(model, loader, device, epoch, params):
    model.eval()
    yaw_err = pitch_err = roll_err = 0.0
    total   = 0

    for imgs, yaw_gt, pitch_gt, roll_gt in loader:
        imgs = imgs.to(device, non_blocking=True)
        bs   = imgs.size(0)
        total += bs

        yaw_pred, pitch_pred, roll_pred = model.predict_angles(imgs)

        yaw_err   += angle_mae(yaw_pred.cpu(),   yaw_gt).sum().item()
        pitch_err += angle_mae(pitch_pred.cpu(), pitch_gt).sum().item()
        roll_err  += angle_mae(roll_pred.cpu(),  roll_gt).sum().item()

    yaw_mae   = yaw_err   / total
    pitch_mae = pitch_err / total
    roll_mae  = roll_err  / total
    mae       = (yaw_mae + pitch_mae + roll_mae) / 3.0

    LOGGER.info(
        f'[Eval Epoch {epoch+1}]  '
        f'Yaw: {yaw_mae:.4f}°  '
        f'Pitch: {pitch_mae:.4f}°  '
        f'Roll: {roll_mae:.4f}°  '
        f'MAE: {mae:.4f}°'
    )
    return mae


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    params = parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f'Device: {device}')

    # ── save dir ──────────────────────────────────────────────────────────────
    save_dir = os.path.join(params.save_path, 'whenet')
    os.makedirs(save_dir, exist_ok=True)

    # ── datasets ──────────────────────────────────────────────────────────────
    train_dir = os.path.join(params.data, '300W_LP')
    val_dir   = os.path.join(params.data, 'AFLW2000')

    train_ds = WHENetPose300W(train_dir, transform=TRAIN_TRANSFORM)
    val_ds   = WHENetAFLW2000(val_dir,   transform=EVAL_TRANSFORM)

    train_loader = DataLoader(
        train_ds,
        batch_size  = params.batch_size,
        shuffle     = True,
        num_workers = params.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = params.batch_size,
        shuffle     = False,
        num_workers = params.num_workers,
        pin_memory  = True,
    )

    LOGGER.info(f'Train: {len(train_ds)} samples, Val: {len(val_ds)} samples')

    # ── model ─────────────────────────────────────────────────────────────────
    pretrained = not params.no_pretrain
    model = WHENet(pretrained=pretrained).to(device)
    LOGGER.info(f'WHENet loaded (pretrained={pretrained})')

    # ── loss / optimizer / scheduler ──────────────────────────────────────────
    criterion  = WHENetLoss(alpha=params.alpha, beta=params.beta)
    optimizer  = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=params.milestones, gamma=params.gamma
    )

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_mae    = float('inf')

    if params.checkpoint and os.path.isfile(params.checkpoint):
        ckpt = torch.load(params.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        best_mae    = ckpt.get('best_mae', float('inf'))
        LOGGER.info(f'Resumed from {params.checkpoint}, epoch {start_epoch}, best MAE {best_mae:.4f}°')

    # ── training loop ─────────────────────────────────────────────────────────
    LOGGER.info('Starting training...')
    for epoch in range(start_epoch, params.epochs):
        train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, params)
        scheduler.step()

        mae = evaluate(model, val_loader, device, epoch, params)

        # save last
        ckpt = {
            'epoch':     epoch + 1,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_mae':  best_mae,
            'args':      params,
        }
        torch.save(ckpt, os.path.join(save_dir, 'last_checkpoint.ckpt'))

        # save best
        if mae < best_mae:
            best_mae = mae
            torch.save(ckpt, os.path.join(save_dir, 'best_checkpoint.ckpt'))
            LOGGER.info(f'★ New best MAE: {best_mae:.4f}°  → saved best_checkpoint.ckpt')

    LOGGER.info(f'Training complete. Best MAE: {best_mae:.4f}°')


if __name__ == '__main__':
    main()
