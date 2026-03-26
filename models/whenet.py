"""
WHENet: Wide-Range Head Pose Estimation Network
PyTorch reimplementation based on the paper architecture:
- EfficientNet-B0 backbone (pretrained ImageNet)
- Bin classification + residual regression (same as Hopenet but wider range for yaw)
- Yaw: 120 bins x 3° = 360° full range
- Pitch/Roll: 66 bins x 3° = ±99°
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class WHENet(nn.Module):
    """
    WHENet: EfficientNet-B0 + bin classification head.

    Yaw:   120 bins, 3°/bin, covers -180° ~ +180°  (full range)
    Pitch:  66 bins, 3°/bin, covers  -99° ~  +99°
    Roll:   66 bins, 3°/bin, covers  -99° ~  +99°
    """

    YAW_BINS   = 120   # 3 * 120 = 360°
    PITCH_BINS =  66   # 3 *  66 = 198° → ±99°
    ROLL_BINS  =  66

    DEG_PER_BIN = 3.0

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ── backbone ──────────────────────────────────────────────────────────
        eff = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # drop the original classifier, keep feature extractor only
        self.backbone = eff.features          # output: (B, 1280, 7, 7) for 224-input
        self.pool     = nn.AdaptiveAvgPool2d(1)  # → (B, 1280)

        feat_dim = 1280  # EfficientNet-B0 feature dimension

        # ── output heads ──────────────────────────────────────────────────────
        self.fc_yaw   = nn.Linear(feat_dim, self.YAW_BINS)
        self.fc_pitch = nn.Linear(feat_dim, self.PITCH_BINS)
        self.fc_roll  = nn.Linear(feat_dim, self.ROLL_BINS)

        # index tensors for expectation computation (registered as buffers)
        self.register_buffer(
            'idx_yaw',
            torch.arange(self.YAW_BINS, dtype=torch.float32)
        )
        self.register_buffer(
            'idx_pitch',
            torch.arange(self.PITCH_BINS, dtype=torch.float32)
        )
        self.register_buffer(
            'idx_roll',
            torch.arange(self.ROLL_BINS, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Returns:
            yaw_logit   (B, 120)
            pitch_logit (B,  66)
            roll_logit  (B,  66)
        """
        feat = self.pool(self.backbone(x))     # (B, 1280, 1, 1)
        feat = feat.flatten(1)                 # (B, 1280)

        yaw_logit   = self.fc_yaw(feat)
        pitch_logit = self.fc_pitch(feat)
        roll_logit  = self.fc_roll(feat)

        return yaw_logit, pitch_logit, roll_logit

    # ------------------------------------------------------------------
    def predict_angles(self, x):
        """
        Convenience: returns predicted angles in degrees (no grad).
        Returns:
            yaw   (B,)  degrees, -180 ~ +180
            pitch (B,)  degrees,  -99 ~ +99
            roll  (B,)  degrees,  -99 ~ +99
        """
        yaw_l, pitch_l, roll_l = self.forward(x)

        yaw   = self._softmax_expectation(yaw_l,   self.idx_yaw)   * self.DEG_PER_BIN - 180.0
        pitch = self._softmax_expectation(pitch_l, self.idx_pitch)  * self.DEG_PER_BIN -  99.0
        roll  = self._softmax_expectation(roll_l,  self.idx_roll)   * self.DEG_PER_BIN -  99.0

        return yaw, pitch, roll

    @staticmethod
    def _softmax_expectation(logits, idx):
        """Weighted expectation: sum(softmax(logits) * idx)."""
        prob = torch.softmax(logits, dim=1)   # (B, num_bins)
        return torch.sum(prob * idx, dim=1)   # (B,)
