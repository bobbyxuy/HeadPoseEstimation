#!/usr/bin/env python3
"""
Head Pose Estimation 推理脚本 — 自动裁剪头部
输入：任意尺寸图片（自动检测人脸并裁剪）
输出：Yaw / Pitch / Roll（度）

用法：
    # 单张图片
    python infer_headpose_crop.py --image test.jpg --output_dir output

    # 目录批量
    python infer_headpose_crop.py --image_dir ./images/ --output_dir output

    # 指定检测方法
    python infer_headpose_crop.py --image test.jpg --detector haar
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_headpose import load_model, infer

import torch


# ── 人脸检测器 ──

HAAR_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
LANDMARKER_PATH = "/tmp/face_landmarker.task"


def detect_faces_haar(img_bgr):
    """OpenCV Haar 级联检测（快但不太准，不需要额外依赖）"""
    cascade = cv2.CascadeClassifier(HAAR_PATH)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]


def detect_faces_landmarker(img_bgr):
    """用 MediaPipe FaceLandmarker 检测人脸（准，支持大角度）"""
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    if not os.path.exists(LANDMARKER_PATH):
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            LANDMARKER_PATH,
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=5,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return []

    h, w = img_bgr.shape[:2]
    boxes = []
    for lm in result.face_landmarks:
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


# ── 裁剪头部 ──


def crop_head(img, bbox, expand_ratio=0.4):
    """
    从检测到的人脸框裁剪头部区域

    Args:
        img: 原始图像
        bbox: (x, y, w, h) 人脸检测框
        expand_ratio: 扩展比例

    Returns:
        crop_img: 裁剪后的头部图像
        crop_bbox: 裁剪框坐标 (x1, y1, x2, y2)
    """
    x, y, bw, bh = bbox
    ih, iw = img.shape[:2]

    expand_lr = int(bw * 0.15)
    expand_h_top = int(bh * 0.5)
    expand_h_bottom = int(bh * 0.15)

    x1 = max(0, x - expand_lr)
    y1 = max(0, y - expand_h_top)
    x2 = min(iw, x + bw + expand_lr)
    y2 = min(ih, y + bh + expand_h_bottom)
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


# ── 主推理逻辑 ──


def process_image(model, img_path, detect_fn, device, output_dir=None):
    """处理单张图片：检测 → 裁剪 → 推理"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[跳过] 无法读取: {img_path}")
        return []

    basename = os.path.basename(img_path)
    results = []

    faces = detect_fn(img)
    if len(faces) == 0:
        print(f"{basename:30s}  未检测到人脸")
        return []

    for i, bbox in enumerate(faces):
        x, y, bw, bh = bbox
        crop, crop_coords = crop_head(img, bbox)
        crop_h, crop_w = crop.shape[:2]

        if crop_h < 30 or crop_w < 30:
            print(f"{basename:30s}  Face[{i}]: 裁剪区域太小，跳过")
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = gray.astype(np.uint8)
        r = infer(model, gray, device)

        results.append({
            "bbox": bbox,
            "crop_coords": crop_coords,
            "yaw": r["yaw"],
            "pitch": r["pitch"],
            "roll": r["roll"],
        })

        label = f'Face[{i}]: Yaw={r["yaw"]:+.1f} Pitch={r["pitch"]:+.1f} Roll={r["roll"]:+.1f}'
        print(f"{basename:30s}  {label}")

    # 保存可视化
    if output_dir and results:
        os.makedirs(output_dir, exist_ok=True)
        vis = img.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX
        stem, ext = os.path.splitext(basename)

        for i, res in enumerate(results):
            x, y, bw, bh = res["bbox"]
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 150, 0), 2)
            cx1, cy1, cx2, cy2 = res["crop_coords"]
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 0), 1)

            lines = [
                f'Yaw:{res["yaw"]:+.1f}',
                f'Pitch:{res["pitch"]:+.1f}',
                f'Roll:{res["roll"]:+.1f}',
            ]
            y0 = y - 10
            for li in lines:
                (tw, _), _ = cv2.getTextSize(li, font, 0.6, 2)
                y0 -= 28
            y0 = max(10, y0 - 5)
            cv2.rectangle(vis, (x - 3, y0 - 25), (x + tw + 5, y0 + 5), (0, 0, 0), -1)
            for j, line in enumerate(lines):
                cv2.putText(vis, line, (x + 3, y0 + j * 25),
                            font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # 保存裁剪图
            crop = crop_head(img, res["bbox"])[0]
            crop_path = os.path.join(output_dir, f"{stem}_face{i}_crop.png")
            cv2.imwrite(crop_path, crop)

        vis_path = os.path.join(output_dir, stem + "_result.png")
        cv2.imwrite(vis_path, vis)
        print(f"  -> {vis_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Head Pose Estimation - 自动裁剪头部推理")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--image_dir", type=str, help="图片目录（批量推理）")
    parser.add_argument("--ckpt", type=str,
                        default="weights/mobilenetv4_medium_pretrained/best_checkpoint.ckpt",
                        help="模型 checkpoint")
    parser.add_argument("--network", type=str, default="mobilenetv4_conv_medium",
                        help="模型架构")
    parser.add_argument("--detector", type=str, default="mediapipe",
                        choices=["haar", "mediapipe"],
                        help="人脸检测方法")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("请指定 --image 或 --image_dir")

    # 初始化模型
    print(f"Loading head pose model ({args.network}) on {args.device}...")
    model = load_model(args.ckpt, args.network, args.device, from_ckpt=True)

    # 设置检测器
    if args.detector == "mediapipe":
        # libGLESv2 软链接
        lib_dir = os.path.expanduser("~/.local/lib")
        lib_gles = os.path.join(lib_dir, "libGLESv2.so.2")
        if os.path.exists("/usr/lib/x86_64-linux-gnu/libGLESv2_nvidia.so.2") and not os.path.exists(lib_gles):
            os.makedirs(lib_dir, exist_ok=True)
            os.symlink("/usr/lib/x86_64-linux-gnu/libGLESv2_nvidia.so.2", lib_gles)
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        detect_fn = detect_faces_landmarker
        print("Detector: MediaPipe FaceLandmarker")
    else:
        detect_fn = detect_faces_haar
        print("Detector: OpenCV Haar Cascade")

    # 收集图片
    images = []
    if args.image:
        images.append(args.image)
    else:
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().rsplit(".", 1)[-1] in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
                images.append(os.path.join(args.image_dir, f))
    if not images:
        print("未找到图片文件")
        return

    print(f"Images: {len(images)}")
    print("-" * 65)

    for img_path in images:
        process_image(model, img_path, detect_fn, args.device, args.output_dir)

    print("-" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
