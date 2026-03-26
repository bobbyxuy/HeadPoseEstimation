"""
Dataset wrappers for WHENet training.
Returns (image, yaw_deg, pitch_deg, roll_deg) instead of rotation matrices.

300W-LP  → training
AFLW2000 → evaluation
"""

import os
import random
import numpy as np
from scipy import io
from PIL import Image, ImageFilter

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scan_mat_files(root_dir, max_abs_angle=99.0, yaw_limit=180.0):
    """Walk root_dir, return list of stem paths that pass angle filters."""
    filenames = []
    skipped   = 0
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            if not f.endswith('.jpg'):
                continue
            stem     = os.path.join(dirpath, f[:-4])
            mat_path = stem + '.mat'
            if not os.path.exists(mat_path):
                continue
            lbl   = io.loadmat(mat_path)
            pitch = float(lbl['Pose_Para'][0][0]) * 180.0 / np.pi
            yaw   = float(lbl['Pose_Para'][0][1]) * 180.0 / np.pi
            roll  = float(lbl['Pose_Para'][0][2]) * 180.0 / np.pi
            if abs(pitch) <= max_abs_angle and abs(yaw) <= yaw_limit and abs(roll) <= max_abs_angle:
                filenames.append(stem)
            else:
                skipped += 1
    return filenames, skipped


# ─────────────────────────────────────────────────────────────────────────────
# 300W-LP  (training)
# ─────────────────────────────────────────────────────────────────────────────

class WHENetPose300W(Dataset):
    """
    300W-LP dataset for WHENet training.

    Label: (yaw_deg, pitch_deg, roll_deg) in degrees.
    Yaw range allowed: ±180° (WHENet handles full-range yaw).
    Pitch/Roll range allowed: ±99°.
    """

    def __init__(self, root: str, transform=None):
        self.root      = root
        self.transform = transform
        self.filenames, n_skip = _scan_mat_files(root, max_abs_angle=99.0, yaw_limit=180.0)
        print(
            f'[WHENetPose300W] Loaded {len(self.filenames)} samples '
            f'(skipped {n_skip} outside angle range).'
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        stem = self.filenames[idx]
        img  = Image.open(stem + '.jpg').convert('RGB')
        lbl  = io.loadmat(stem + '.mat')

        pitch = float(lbl['Pose_Para'][0][0]) * 180.0 / np.pi
        yaw   = float(lbl['Pose_Para'][0][1]) * 180.0 / np.pi
        roll  = float(lbl['Pose_Para'][0][2]) * 180.0 / np.pi

        # ── crop face region (same as existing Pose300W) ──────────────────
        pt2d  = lbl['pt2d']
        x_min = np.min(pt2d[0])
        x_max = np.max(pt2d[0])
        y_min = np.min(pt2d[1])
        y_max = np.max(pt2d[1])

        k  = random.uniform(0.2, 0.4)
        dx = 0.6 * k * (x_max - x_min)
        dy = 0.6 * k * (y_max - y_min)
        x_min -= dx;  x_max += dx
        y_min -= 2 * dy;  y_max += dy
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # ── augmentation ──────────────────────────────────────────────────
        if random.random() < 0.5:          # horizontal flip
            yaw  = -yaw
            roll = -roll
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.05:         # slight blur
            img = img.filter(ImageFilter.BLUR)

        if self.transform:
            img = self.transform(img)

        return (
            img,
            torch.tensor(yaw,   dtype=torch.float32),
            torch.tensor(pitch, dtype=torch.float32),
            torch.tensor(roll,  dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# AFLW2000  (evaluation)
# ─────────────────────────────────────────────────────────────────────────────

class WHENetAFLW2000(Dataset):
    """
    AFLW2000 dataset for WHENet evaluation.
    Same angle filters as training (pitch/roll ≤99°, yaw ≤180°).
    """

    def __init__(self, root: str, transform=None):
        self.root      = root
        self.transform = transform
        self.filenames, n_skip = _scan_mat_files(root, max_abs_angle=99.0, yaw_limit=180.0)
        print(
            f'[WHENetAFLW2000] Loaded {len(self.filenames)} samples '
            f'(skipped {n_skip} outside angle range).'
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        stem = self.filenames[idx]
        img  = Image.open(stem + '.jpg').convert('RGB')
        lbl  = io.loadmat(stem + '.mat')

        pitch = float(lbl['Pose_Para'][0][0]) * 180.0 / np.pi
        yaw   = float(lbl['Pose_Para'][0][1]) * 180.0 / np.pi
        roll  = float(lbl['Pose_Para'][0][2]) * 180.0 / np.pi

        # ── crop face region ──────────────────────────────────────────────
        pt2d  = lbl['pt2d']
        x_min = np.min(pt2d[0])
        x_max = np.max(pt2d[0])
        y_min = np.min(pt2d[1])
        y_max = np.max(pt2d[1])

        x_min -= 0.2 * (x_max - x_min)
        x_max += 0.2 * (x_max - x_min)
        y_min -= 0.2 * (y_max - y_min)
        y_max += 0.2 * (y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        if self.transform:
            img = self.transform(img)

        return (
            img,
            torch.tensor(yaw,   dtype=torch.float32),
            torch.tensor(pitch, dtype=torch.float32),
            torch.tensor(roll,  dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# transforms
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
