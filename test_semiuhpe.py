"""
SemiUHPE inference with BPJDet head detection and visualization.

Mode 1 (--mode scene): Full scene images -> BPJDet detects heads -> crop -> SemiUHPE predicts pose
Mode 2 (--mode crop):  Pre-cropped head images -> SemiUHPE predicts pose directly

Outputs annotated images with crop region, 3D axis, and euler angles.

==========================================================================
Model Download Instructions
==========================================================================

All weights should be placed under:  head-pose-estimation/weights/

1) SemiUHPE (EfficientNetV2-S) - WildHead dataset (recommended):
   wget -c https://huggingface.co/HerryChou/SemiUHPE/resolve/main/DAD-WildHead-EffNetV2-S-best.pth \
         -O weights/DAD-WildHead-EffNetV2-S-best.pth

2) SemiUHPE (EfficientNetV2-S) - COCOHead dataset:
   wget -c https://huggingface.co/HerryChou/SemiUHPE/resolve/main/DAD-COCOHead-EffNetV2-S-best.pth \
         -O weights/DAD-COCOHead-EffNetV2-S-best.pth

3) BPJDet head detector (required for --mode scene only):
   wget -c https://huggingface.co/HerryChou/BPJDet/resolve/main/ch_head_l_1536_e150_best_mMR.pt \
         -O weights/ch_head_l_1536_e150_best_mMR.pt

If download is slow, try using a proxy (e.g. port 7890):
   export https_proxy=http://127.0.0.1:7890
   wget -c <url>

Dependencies:
   pip install torch torchvision opencv-python scipy pillow

SemiUHPE code (needed for BPJDet imports in scene mode):
   git clone https://github.com/HerryChou/SemiUHPE.git ../SemiUHPE
   (Must be cloned as a sibling directory: head-pose-estimation/../SemiUHPE/)

Note: If using PyTorch >= 2.6, patch SemiUHPE/models/experimental.py line 94:
   torch.load(attempt_download(w), map_location=map_location)
   -> torch.load(attempt_download(w), map_location=map_location, weights_only=False)
==========================================================================
"""
import sys
import os
import cv2
import glob
import time
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.models as models
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import argparse

# Add SemiUHPE project for BPJDet dependencies (must be FIRST to avoid conflicts)
SEMIUHPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SemiUHPE')
sys.path.insert(0, SEMIUHPE_DIR)
# Remove current dir from sys.path to avoid model name conflicts
if os.path.dirname(os.path.abspath(__file__)) in sys.path:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))


def build_semiuhpe_effinetv2s(num_classes=9):
    """Build EfficientNetV2-S with the same classifier head as SemiUHPE."""
    model = models.efficientnet_v2_s(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU6(inplace=True),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU6(inplace=True),
        nn.Linear(128, num_classes),
    )
    return model


def load_semiuhpe_checkpoint(model, ckpt_path, use_ema=True):
    """Load SemiUHPE checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    key = 'model_state_dict_ema' if (use_ema and 'model_state_dict_ema' in ckpt) else 'model_state_dict'
    model.load_state_dict(ckpt[key])
    print(f"Loaded {'EMA ' if use_ema else ''}weights from {ckpt_path}")
    return model


def fisher_A_to_R(A):
    """Convert matrix Fisher parameter A (9-dim) to rotation matrix R via SVD."""
    A = A.reshape(-1, 3, 3)
    U, S, V = torch.svd(A)
    with torch.no_grad():
        s3sign = torch.det(torch.matmul(U, V.transpose(1, 2)))
    U = torch.cat((U[:, :, :2], U[:, :, 2:] * s3sign[:, None][:, None]), -1)
    R = torch.matmul(U, V.transpose(1, 2))
    return R


def limit_angle(angle, pi=180.0):
    if angle < -pi:
        k = -2 * (int(angle / pi) // 2)
        angle = angle + k * pi
    if angle > pi:
        k = 2 * ((int(angle / pi) + 1) // 2)
        angle = angle - k * pi
    return angle


def draw_axis_ypr(img, yaw, pitch, roll, tdx=None, tdy=None, size=80):
    """Draw 3D axis on image. Referenced from HopeNet."""
    pitch_rad = pitch * np.pi / 180
    yaw_rad = -(yaw * np.pi / 180)
    roll_rad = roll * np.pi / 180

    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis (red), Y-Axis (green), Z-Axis (blue)
    x1 = size * (np.cos(yaw_rad) * np.cos(roll_rad)) + tdx
    y1 = size * (np.cos(pitch_rad) * np.sin(roll_rad) + np.cos(roll_rad) * np.sin(pitch_rad) * np.sin(yaw_rad)) + tdy
    x2 = size * (-np.cos(yaw_rad) * np.sin(roll_rad)) + tdx
    y2 = size * (np.cos(pitch_rad) * np.cos(roll_rad) - np.sin(pitch_rad) * np.sin(yaw_rad) * np.sin(roll_rad)) + tdy
    x3 = size * (np.sin(yaw_rad)) + tdx
    y3 = size * (-np.cos(yaw_rad) * np.sin(pitch_rad)) + tdy

    thickness = max(3, img.shape[0] // 80)
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), thickness)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), thickness)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), thickness)
    return img


def load_bpjdet(weights_path, device='cpu'):
    """Load BPJDet head detection model."""
    from models.experimental import attempt_load
    model = attempt_load(weights_path, map_location=device)
    print(f"BPJDet loaded from {weights_path}")
    return model


def detect_heads_bpjdet(bpjdet_model, img_bgr, device='cpu', conf_thres=0.5, iou_thres=0.75, imgsz=1536):
    """Detect heads using BPJDet. Returns list of head bboxes [x1,y1,x2,y2]."""
    from utils.general import check_img_size, non_max_suppression, scale_coords

    stride = int(bpjdet_model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    img = cv2.resize(img_bgr, (imgsz, imgsz))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if len(img.shape) == 3:
        img = img[None]

    with torch.no_grad():
        out = bpjdet_model(img, augment=True)[0]

    num_offsets = 2
    body_dets = non_max_suppression(out, conf_thres, iou_thres, classes=[0], num_offsets=num_offsets)
    part_dets = non_max_suppression(out, conf_thres, iou_thres, classes=list(range(1, 1 + num_offsets // 2)), num_offsets=num_offsets)

    orig_shape = img_bgr.shape[:2]
    head_bboxes = []

    for bdet, pdet in zip(body_dets, part_dets):
        if bdet.shape[0] == 0:
            continue

        bdet[:, :4] = scale_coords(img.shape[2:], bdet[:, :4], orig_shape).round()
        bboxes_np = bdet[:, :4].cpu().numpy()

        if pdet.shape[0] > 0:
            pdet[:, :4] = scale_coords(img.shape[2:], pdet[:, :4].clone(), orig_shape)
            pdet_slim = pdet[:, :6].cpu().numpy()
        else:
            pdet_slim = np.empty((0, 6))

        body_head_pairs = {}
        for x1, y1, x2, y2, conf, cls in pdet_slim:
            p_xc, p_yc = np.mean((x1, x2)), np.mean((y1, y2))
            body_centers = np.stack([
                (bboxes_np[:, 0] + bboxes_np[:, 2]) / 2,
                (bboxes_np[:, 1] + bboxes_np[:, 3]) / 2
            ], axis=-1)
            dist = np.linalg.norm(body_centers - np.array([[p_xc, p_yc]]), axis=-1)
            pt_match = np.argmin(dist)
            body_key = int(pt_match)
            if body_key not in body_head_pairs or conf > body_head_pairs[body_key][4]:
                body_head_pairs[body_key] = [x1, y1, x2, y2, conf]

        for idx, (x1, y1, x2, y2, conf) in body_head_pairs.items():
            head_bboxes.append([int(x1), int(y1), int(x2), int(y2)])

    return head_bboxes


def crop_head_bpjdet(img_bgr, head_bbox, edges_scale=-0.05):
    """Crop head using SemiUHPE's aspect-ratio invariant cropping."""
    px1, py1, px2, py2 = head_bbox
    img_h, img_w = img_bgr.shape[:2]

    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    head_size = max(px2 - px1, py2 - py1)
    new_px1 = max(0, int(pcx - (0.5 - edges_scale) * head_size))
    new_px2 = min(img_w - 1, int(pcx + (0.5 - edges_scale) * head_size))
    new_py1 = max(0, int(pcy - (0.5 - edges_scale) * head_size))
    new_py2 = min(img_h - 1, int(pcy + (0.5 - edges_scale) * head_size))

    crop = img_bgr[new_py1:new_py2, new_px1:new_px2]
    return crop, [new_px1, new_py1, new_px2, new_py2]


def predict_pose(model, crop_bgr, transform, device):
    """Run SemiUHPE inference on a cropped head image. Returns (pitch, yaw, roll, elapsed_ms)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    img_tensor = transform(crop_pil).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        fisher_out = model(img_tensor)
        rot_mat = fisher_A_to_R(fisher_out)
    elapsed = (time.time() - start) * 1000

    # Convert to euler angles (DAD3DHeads full-range convention)
    rot_mat_np = rot_mat.detach().cpu().numpy()[0]
    rot_mat_t = np.transpose(rot_mat_np)
    angle = Rotation.from_matrix(rot_mat_t).as_euler("xyz", degrees=True)
    roll = limit_angle(angle[2])
    pitch = limit_angle(angle[0] - 180)
    yaw = limit_angle(angle[1])

    return pitch, yaw, roll, elapsed


def process_scene_images(args, model, bpjdet_model, transform, device):
    """Process full scene images with BPJDet head detection."""
    output_dir = args.output_dir + "_scene"
    os.makedirs(output_dir, exist_ok=True)

    test_images = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))[:args.num_images]
    test_images += sorted(glob.glob(os.path.join(args.image_dir, "*.png")))[:args.num_images]
    test_images = list(dict.fromkeys(test_images))[:args.num_images]

    if not test_images:
        print("No test images found!")
        return

    print(f"\n[Scene Mode] Processing {len(test_images)} images with BPJDet + SemiUHPE...")
    print("=" * 80)

    bpjdet_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    total_inference = 0

    for img_path in test_images:
        img_name = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_vis = img_bgr.copy()
        img_h, img_w = img_bgr.shape[:2]

        head_bboxes = detect_heads_bpjdet(bpjdet_model, img_bgr, device=bpjdet_device, imgsz=args.imgsz)

        if not head_bboxes:
            cv2.putText(img_vis, 'No head detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(output_dir, img_name), img_vis)
            print(f"  {img_name}: No heads detected")
            continue

        for idx, head_bbox in enumerate(head_bboxes):
            hx1, hy1, hx2, hy2 = head_bbox
            cv2.rectangle(img_vis, (hx1, hy1), (hx2, hy2), (0, 255, 255), 3)

            crop, crop_bbox = crop_head_bpjdet(img_bgr, head_bbox)
            if crop.size == 0:
                continue

            cx1, cy1, cx2, cy2 = crop_bbox
            cv2.rectangle(img_vis, (cx1, cy1), (cx2, cy2), (255, 255, 0), 3)

            pitch, yaw, roll, elapsed = predict_pose(model, crop, transform, device)
            total_inference += elapsed

            hcx = (hx1 + hx2) // 2
            hcy = (hy1 + hy2) // 2
            axis_size = max(40, (hx2 - hx1) // 2)
            draw_axis_ypr(img_vis, yaw, pitch, roll, tdx=hcx, tdy=hcy, size=axis_size)

            text = f"P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}"
            font_scale = max(0.8, img_h / 800)
            thickness = max(2, int(font_scale * 2))
            cv2.putText(img_vis, text, (hx1, max(30, hy1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            print(f"  {img_name} head[{idx}]: P={pitch:7.2f}, Y={yaw:7.2f}, R={roll:7.2f}  ({elapsed:.1f}ms)")

        cv2.putText(img_vis, f'{len(head_bboxes)} Head(s)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, img_vis)
        print(f"  Saved: {save_path}")

    print("=" * 80)
    print(f"Results saved to {output_dir}/")


def process_crop_images(args, model, transform, device):
    """Process pre-cropped head images directly with SemiUHPE."""
    output_dir = args.output_dir + "_crop"
    os.makedirs(output_dir, exist_ok=True)

    test_images = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))[:args.num_images]
    test_images += sorted(glob.glob(os.path.join(args.image_dir, "*.png")))[:args.num_images]
    test_images = list(dict.fromkeys(test_images))[:args.num_images]

    if not test_images:
        print("No test images found!")
        return

    print(f"\n[Crop Mode] Processing {len(test_images)} cropped head images with SemiUHPE...")
    print("=" * 80)

    total_inference = 0

    for img_path in test_images:
        img_name = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_vis = img_bgr.copy()
        img_h, img_w = img_bgr.shape[:2]

        # Enlarge the visualization canvas for better annotation
        vis_scale = max(1, 600 // min(img_h, img_w))
        img_vis_large = cv2.resize(img_vis, (img_w * vis_scale, img_h * vis_scale))

        pitch, yaw, roll, elapsed = predict_pose(model, img_bgr, transform, device)
        total_inference += elapsed

        # Draw axis at center of enlarged image
        cx = img_w * vis_scale // 2
        cy = img_h * vis_scale // 2
        axis_size = min(img_w, img_h) * vis_scale // 3
        draw_axis_ypr(img_vis_large, yaw, pitch, roll, tdx=cx, tdy=cy, size=axis_size)

        # Draw euler angles text
        text = f"Pitch:{pitch:.1f}  Yaw:{yaw:.1f}  Roll:{roll:.1f}"
        font_scale = max(0.8, img_vis_large.shape[0] / 400)
        thickness = max(2, int(font_scale * 2.5))
        cv2.putText(img_vis_large, text, (10, img_vis_large.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # Draw original crop border
        cv2.rectangle(img_vis_large, (0, 0),
                      (img_w * vis_scale - 1, img_h * vis_scale - 1), (255, 255, 0), 3)

        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, img_vis_large)
        print(f"  {img_name}: P={pitch:7.2f}, Y={yaw:7.2f}, R={roll:7.2f}  ({elapsed:.1f}ms)  Saved")

    print("=" * 80)
    print(f"Results saved to {output_dir}/")
    print(f"Total inference time: {total_inference:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description='SemiUHPE inference test')
    parser.add_argument('--mode', type=str, default='crop', choices=['scene', 'crop'],
                        help='scene: full images with BPJDet detection; crop: pre-cropped head images')
    parser.add_argument('--image_dir', type=str, default='data/AFLW2000',
                        help='directory of test images')
    parser.add_argument('--output_dir', type=str, default='output_semiuhpe',
                        help='output directory for annotated images')
    parser.add_argument('--weights', type=str, default='weights/DAD-WildHead-EffNetV2-S-best.pth',
                        help='SemiUHPE model weights')
    parser.add_argument('--bpjdet_weights', type=str, default='weights/ch_head_l_1536_e150_best_mMR.pt',
                        help='BPJDet head detection weights')
    parser.add_argument('--num_images', type=int, default=10,
                        help='number of images to test')
    parser.add_argument('--imgsz', type=int, default=1536,
                        help='BPJDet inference image size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build SemiUHPE model
    model = build_semiuhpe_effinetv2s(num_classes=9)
    model = load_semiuhpe_checkpoint(model, args.weights, use_ema=True)
    model = model.to(device).eval()

    transform = tfs.Compose([
        tfs.Resize((224, 224)),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.mode == 'scene':
        bpjdet_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        bpjdet_model = load_bpjdet(args.bpjdet_weights, device=bpjdet_device)
        process_scene_images(args, model, bpjdet_model, transform, device)
    else:
        process_crop_images(args, model, transform, device)


if __name__ == '__main__':
    main()
