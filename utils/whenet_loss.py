"""
WHENet Loss: Bin Classification + Residual Regression
Same formulation as Hopenet (CVPR 2017), extended to ±180° for yaw.

For each angle (yaw/pitch/roll):
  loss = alpha * CE(logits, bin_label) + (1-alpha) * MSE(expectation, gt_angle_norm)

where:
  bin_label     = floor((gt_deg + offset) / deg_per_bin)   integer class label
  expectation   = sum(softmax(logits) * idx)                soft predicted bin
  gt_angle_norm = (gt_deg + offset) / deg_per_bin          normalised gt
  alpha         = weight for classification loss (default 1.0, regression 2.0 per paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WHENetLoss(nn.Module):
    """
    Combined bin-classification + soft-regression loss for WHENet.

    Args:
        alpha (float): weight for cross-entropy classification loss.
        beta  (float): weight for MSE regression loss.
    """

    # match WHENet architecture
    YAW_BINS    = 120    # yaw:   -180 ~ +177° in 3° steps
    PITCH_BINS  =  66    # pitch:  -99 ~  +96° in 3° steps
    ROLL_BINS   =  66    # roll:   -99 ~  +96° in 3° steps
    DEG_PER_BIN =   3.0
    YAW_OFFSET  = 180.0  # shift so 0° → bin 60
    PR_OFFSET   =  99.0  # shift so 0° → bin 33

    def __init__(self, alpha: float = 1.0, beta: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.ce    = nn.CrossEntropyLoss()
        self.mse   = nn.MSELoss()

    # ------------------------------------------------------------------
    def _angle_to_bin(self, angle_deg: torch.Tensor, offset: float, num_bins: int) -> torch.Tensor:
        """Convert angle in degrees to integer bin label (clipped to valid range)."""
        bins = ((angle_deg + offset) / self.DEG_PER_BIN).long()
        return bins.clamp(0, num_bins - 1)

    def _angle_to_norm(self, angle_deg: torch.Tensor, offset: float) -> torch.Tensor:
        """Convert angle to normalised bin index (float, for regression target)."""
        return (angle_deg + offset) / self.DEG_PER_BIN

    @staticmethod
    def _softmax_expectation(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)
        return torch.sum(prob * idx, dim=1)

    # ------------------------------------------------------------------
    def forward(
        self,
        yaw_logit:   torch.Tensor,   # (B, 120)
        pitch_logit: torch.Tensor,   # (B,  66)
        roll_logit:  torch.Tensor,   # (B,  66)
        yaw_gt:      torch.Tensor,   # (B,)  degrees
        pitch_gt:    torch.Tensor,   # (B,)  degrees
        roll_gt:     torch.Tensor,   # (B,)  degrees
    ):
        device = yaw_logit.device

        # ── index tensors ────────────────────────────────────────────────────
        idx_yaw   = torch.arange(self.YAW_BINS,   dtype=torch.float32, device=device)
        idx_pitch = torch.arange(self.PITCH_BINS, dtype=torch.float32, device=device)
        idx_roll  = torch.arange(self.ROLL_BINS,  dtype=torch.float32, device=device)

        # ── bin labels (classification targets) ──────────────────────────────
        yaw_bin   = self._angle_to_bin(yaw_gt,   self.YAW_OFFSET, self.YAW_BINS)
        pitch_bin = self._angle_to_bin(pitch_gt, self.PR_OFFSET,  self.PITCH_BINS)
        roll_bin  = self._angle_to_bin(roll_gt,  self.PR_OFFSET,  self.ROLL_BINS)

        # ── classification loss ───────────────────────────────────────────────
        ce_yaw   = self.ce(yaw_logit,   yaw_bin)
        ce_pitch = self.ce(pitch_logit, pitch_bin)
        ce_roll  = self.ce(roll_logit,  roll_bin)

        # ── soft regression: predicted expectation vs normalised gt ───────────
        exp_yaw   = self._softmax_expectation(yaw_logit,   idx_yaw)
        exp_pitch = self._softmax_expectation(pitch_logit, idx_pitch)
        exp_roll  = self._softmax_expectation(roll_logit,  idx_roll)

        norm_yaw   = self._angle_to_norm(yaw_gt,   self.YAW_OFFSET)
        norm_pitch = self._angle_to_norm(pitch_gt, self.PR_OFFSET)
        norm_roll  = self._angle_to_norm(roll_gt,  self.PR_OFFSET)

        mse_yaw   = self.mse(exp_yaw,   norm_yaw)
        mse_pitch = self.mse(exp_pitch, norm_pitch)
        mse_roll  = self.mse(exp_roll,  norm_roll)

        # ── combine ───────────────────────────────────────────────────────────
        loss = (
            self.alpha * (ce_yaw + ce_pitch + ce_roll) +
            self.beta  * (mse_yaw + mse_pitch + mse_roll)
        )
        return loss, ce_yaw + ce_pitch + ce_roll, mse_yaw + mse_pitch + mse_roll
