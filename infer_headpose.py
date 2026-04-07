#!/usr/bin/env python3
"""
Head Pose Estimation 推理脚本
输入：灰度图片（任意尺寸，如 640x480）
输出：Yaw / Pitch / Roll（度）

用法：
    python infer_headpose.py --image test.png --weights weights/mobilenetv4_conv_medium_pretrained.pth --network mobilenetv4_conv_medium
    python infer_headpose.py --image test.png --ckpt weights/mobilenetv4_medium_pretrained/best_checkpoint.ckpt --network mobilenetv4_conv_medium
    python infer_headpose.py --image_dir ./images/ --weights weights/mobilenetv4_conv_medium_pretrained.pth --network mobilenetv4_conv_medium --output_json
"""

import os
import sys
import argparse
import json
import warnings

import cv2
import numpy as np
import scipy.io as sio
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

# ── Preprocessing ──────────────────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess(gray_image):
    """灰度图 (H, W) uint8 → tensor (1, 3, 224, 224)"""
    if gray_image.ndim == 2:
        img_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)
    return _transform(img_rgb).unsqueeze(0)


# ── Inference ──────────────────────────────────────────────────────────────

def infer(model, gray_image, device="cpu"):
    """单张灰度图 → {"yaw": float, "pitch": float, "roll": float} (度)"""
    from utils.general import compute_euler_angles_from_rotation_matrices

    tensor = preprocess(gray_image).to(device)
    with torch.no_grad():
        rotation_matrix = model(tensor)
        euler = np.degrees(
            compute_euler_angles_from_rotation_matrices(rotation_matrix).cpu().numpy()
        )
    return {
        "yaw":   float(euler[0, 1]),
        "pitch": float(euler[0, 0]),
        "roll":  float(euler[0, 2]),
    }


# ── Model Loader ───────────────────────────────────────────────────────────

ARCH_TO_TIMM = {
    "mobilenetv4_conv_small":       "mobilenetv4_conv_small",
    "mobilenetv4_conv_medium":      "mobilenetv4_conv_medium",
    "mobilenetv4_conv_large":       "mobilenetv4_conv_large",
    "mobilenetv4_hybrid_medium":    "mobilenetv4_hybrid_medium",
}

_ARCH_ALIAS = {
    "mobilenetv4_small_pretrained":       "mobilenetv4_conv_small",
    "mobilenetv4_medium_pretrained":      "mobilenetv4_conv_medium",
    "mobilenetv4_large_pretrained":       "mobilenetv4_conv_large",
    "mobilenetv4_hybrid_medium_pretrained": "mobilenetv4_hybrid_medium",
}


def _get_timm_name(arch):
    """解析 timm 模型名"""
    if arch in ARCH_TO_TIMM:
        return ARCH_TO_TIMM[arch]
    if arch in _ARCH_ALIAS:
        return _ARCH_ALIAS[arch]
    # 直接作为 timm 名
    return arch


def load_model(weights_path, arch, device="cpu", from_ckpt=False):
    """加载模型。使用项目自带模型定义，支持 .pth 和 .ckpt。"""
    from models import get_model

    # 映射 CLI 参数到 get_model 支持的 arch 名
    _ARCH_MAP = {
        "mobilenetv4_conv_small":    "mobilenetv4_small_pretrained",
        "mobilenetv4_conv_medium":   "mobilenetv4_medium_pretrained",
        "mobilenetv4_conv_large":    "mobilenetv4_large_pretrained",
        "mobilenetv4_hybrid_medium": "mobilenetv4_hybrid_medium_pretrained",
    }
    model_arch = _ARCH_MAP.get(arch, arch)

    # 构建模型
    model = get_model(model_arch, num_classes=6, pretrained=False)

    # 加载权重
    if from_ckpt:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = ckpt["model"]
    else:
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)

    # 清理不需要的 key
    clean_sd = {}
    for k, v in sd.items():
        if "classifier" in k:
            continue
        clean_sd[k] = v

    model.load_state_dict(clean_sd, strict=False)
    model.to(device)
    model.eval()
    return model


def load_gt_angles(img_path):
    """尝试从同名 .mat 文件读取 GT 角度（300W_LP 数据集格式）。
    返回 {"yaw": float, "pitch": float, "roll": float} (度) 或 None。
    """
    mat_path = os.path.splitext(img_path)[0] + ".mat"
    if not os.path.exists(mat_path):
        return None
    try:
        lbl = sio.loadmat(mat_path)
        pose = lbl["Pose_Para"][0][:3]  # [pitch, yaw, roll] 弧度
        return {
            "pitch": float(pose[0]) * 180.0 / np.pi,
            "yaw":   float(pose[1]) * 180.0 / np.pi,
            "roll":  float(pose[2]) * 180.0 / np.pi,
        }
    except Exception:
        return None


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Head Pose Estimation - 灰度图推理")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--image_dir", type=str, help="图片目录（批量推理）")
    parser.add_argument("--weights", type=str, help="模型权重 (.pth，需含 fc head)")
    parser.add_argument("--ckpt", type=str, help="训练 checkpoint (.ckpt，推荐)")
    parser.add_argument("--network", type=str, default="mobilenetv4_conv_medium",
                        help="模型架构 (default: mobilenetv4_conv_medium)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    parser.add_argument("--output_json", action="store_true", help="JSON 输出")
    parser.add_argument("--output_dir", type=str, help="输出目录：将角度标注绘制在图像上并保存")
    args = parser.parse_args()

    if not args.weights and not args.ckpt:
        parser.error("请指定 --weights 或 --ckpt")
    if not args.image and not args.image_dir:
        parser.error("请指定 --image 或 --image_dir")

    device = torch.device(args.device)

    if args.ckpt:
        model = load_model(args.ckpt, args.network, device, from_ckpt=True)
        weights_label = args.ckpt
    else:
        model = load_model(args.weights, args.network, device, from_ckpt=False)
        weights_label = args.weights

    print(f"模型: {args.network}")
    print(f"权重: {weights_label}")
    print(f"设备: {device}")
    print("-" * 65)

    images = []
    if args.image:
        images.append(args.image)
    else:
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().rsplit(".", 1)[-1] in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
                images.append(os.path.join(args.image_dir, f))

    results = {}
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for img_path in images:
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue
        h, w = gray.shape
        r = infer(model, gray, device)
        results[os.path.basename(img_path)] = r
        gt = load_gt_angles(img_path) if args.output_dir else None
        if args.output_json:
            entry = {"pred": r}
            if gt:
                entry["gt"] = gt
            print(json.dumps({os.path.basename(img_path): entry}))
        else:
            line = (f"{os.path.basename(img_path):30s}  [{w}x{h}]  "
                    f"Yaw={r['yaw']:+7.2f}°  Pitch={r['pitch']:+7.2f}°  Roll={r['roll']:+7.2f}°")
            if gt:
                line += (f"  | GT: Yaw={gt['yaw']:+7.2f}°  Pitch={gt['pitch']:+7.2f}°  Roll={gt['roll']:+7.2f}°")
            print(line)

        # 保存带角度标注的图像
        if args.output_dir:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            gt = load_gt_angles(img_path)

            pred_lines = [
                f"Pred Yaw:   {r['yaw']:+.2f}",
                f"Pred Pitch: {r['pitch']:+.2f}",
                f"Pred Roll:  {r['roll']:+.2f}",
            ]
            if gt:
                pred_lines += [
                    "",
                    f"GT   Yaw:   {gt['yaw']:+.2f}",
                    f"GT   Pitch: {gt['pitch']:+.2f}",
                    f"GT   Roll:  {gt['roll']:+.2f}",
                ]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            line_height = 25
            x0, y0 = 10, 25
            # 背景
            all_text = [l for l in pred_lines if l]
            (tw, th), _ = cv2.getTextSize(max(all_text, key=len), font, font_scale, thickness)
            cv2.rectangle(vis, (x0 - 5, y0 - th - 5),
                          (x0 + tw + 5, y0 + line_height * (len(pred_lines) - 1) + 5),
                          (0, 0, 0), -1)
            for i, line in enumerate(pred_lines):
                if not line:
                    continue
                color = (0, 255, 0) if i < 3 else (0, 200, 255)  # Pred=绿, GT=黄
                cv2.putText(vis, line, (x0, y0 + i * line_height),
                            font, font_scale, color, thickness, cv2.LINE_AA)

            stem, _ = os.path.splitext(os.path.basename(img_path))
            out_path = os.path.join(args.output_dir, stem + ".png")
            cv2.imwrite(out_path, vis)
            print(f"  -> 已保存: {out_path}")

    if args.output_json and len(results) > 1:
        print("\n--- All Results ---")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
