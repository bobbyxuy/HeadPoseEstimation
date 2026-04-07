#!/usr/bin/env python3
"""
MediaPipe 3D Face Mesh → 平滑深度图 → 逆映射视差扭曲
使用 scipy 插值创建平滑深度图，逆向映射避免空洞
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import urllib.request

MODEL_PATH = "/tmp/face_landmarker.task"


def get_face_landmarker():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_face(landmarker, img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    h, w = img_bgr.shape[:2]
    pts = np.zeros((len(lm), 3))
    for i, p in enumerate(lm):
        pts[i] = [p.x * w, p.y * h, p.z * w]
    return pts


def warp_perspective_smooth(img_bgr, pts_3d, cam_offset_x=0.0):
    """
    用平滑深度图做逆映射视差扭曲

    原理：
      1. 用 MediaPipe 3D 关键点 + scipy 插值创建平滑深度图
      2. 对输出图每个面部像素，通过视差逆映射找源像素
      3. 用 cv2.remap 做双线性插值，无空洞
    """
    h, w = img_bgr.shape[:2]

    if cam_offset_x == 0:
        return img_bgr.copy()

    # ── 1. 面部 mask（凸包）──
    hull = ConvexHull(pts_3d[:, :2])
    hull_pts = pts_3d[hull.vertices, :2].astype(np.int32)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull_pts, 255)
    face_mask = face_mask > 0

    # 稍微膨胀 mask 以覆盖边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    face_mask_dilated = cv2.dilate(face_mask.astype(np.uint8), kernel) > 0

    # ── 2. 平滑深度图 ──
    vx = pts_3d[:, 0]
    vy = pts_3d[:, 1]
    vz = -pts_3d[:, 2]  # 翻转：正=远离相机

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # 用 cubic 插值
    depth_interp = griddata(
        (vx, vy), vz,
        (grid_x, grid_y),
        method='cubic',
        fill_value=np.nan
    )

    # 在面部区域填充 NaN（边缘可能没有插值）
    # 用最近邻插值补充
    depth_nearest = griddata(
        (vx, vy), vz,
        (grid_x, grid_y),
        method='nearest',
        fill_value=0
    )

    nan_mask = np.isnan(depth_interp) & face_mask_dilated
    depth_interp[nan_mask] = depth_nearest[nan_mask]

    # 非面部区域设为 0
    depth_interp[~face_mask_dilated] = 0

    # ── 3. 计算视差 ──
    # 焦距（与图像宽度相关）
    f = w * 1.2

    # 将相对深度转为绝对深度
    valid_depth = depth_interp[face_mask_dilated]
    d_min, d_max = valid_depth.min(), valid_depth.max()
    d_range = d_max - d_min if d_max > d_min else 1.0

    face_w = pts_3d[:, 0].max() - pts_3d[:, 0].min()
    base_z = f * 1.0
    z_variation = face_w * 0.4

    depth_abs = np.where(face_mask_dilated,
        base_z - (depth_interp - d_min) / d_range * z_variation,
        base_z)

    # 视差 = f * offset / depth
    disparity = f * cam_offset_x / depth_abs

    # ── 4. 逆映射：构建 map_x, map_y ──
    # 输出像素 (u_out, v_out) 对应的输入像素 (u_src, v_src)
    # 正向: u_out = u_src - disparity(u_src)
    # 逆映射近似: u_src ≈ u_out + disparity(u_out)  (一阶近似)
    # 更精确: 迭代一次
    #   u_src_0 = u_out + disparity_at(u_out)
    #   u_src_1 = u_out + disparity_at(u_src_0)

    map_x = np.arange(w, dtype=np.float32).reshape(1, -1).repeat(h, axis=0).copy()
    map_y = np.arange(h, dtype=np.float32).reshape(-1, 1).repeat(w, axis=1).copy()

    # 对面部区域做视差补偿
    # 第一轮：用输出位置的 disparity 估算源位置
    u_src_est = map_x + disparity.astype(np.float32)

    # 第二轮：用估算的源位置重新计算 disparity（更精确）
    # 采样 disparity at u_src_est
    u_src_int = np.clip(np.round(u_src_est).astype(np.int32), 0, w - 1)
    disparity_refined = disparity[u_src_int, map_y.astype(np.int32)]
    u_src_final = map_x + disparity_refined

    # 面部区域使用逆映射
    map_x[face_mask_dilated] = u_src_final[face_mask_dilated]

    # ── 5. cv2.remap ──
    result = cv2.remap(img_bgr, map_x, map_y,
                       cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Smooth depth warp - A-pillar simulation')
    parser.add_argument('--data_dir', default='data/300W_LP/AFW')
    parser.add_argument('--base', default='AFW_1130084326_3')
    parser.add_argument('--output_dir', default='output_yaw_sweep')
    parser.add_argument('--img_size', type=int, default=450)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Initializing MediaPipe...")
    landmarker = get_face_landmarker()

    print("Loading head pose model...")
    from infer_headpose import load_model, infer, load_gt_angles
    model = load_model('weights/mobilenetv4_medium_pretrained/best_checkpoint.ckpt',
                        'mobilenetv4_conv_medium', 'cpu', from_ckpt=True)

    cam_configs = [
        (0,   'Frontal'),
        (40,  'A-pillar 40'),
        (80,  'A-pillar 80'),
        (130, 'A-pillar 130'),
    ]

    TARGET_H = 220
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    all_rows = []
    all_data = []

    for offset, cam_label in cam_configs:
        items_vis = []
        for idx in range(18):
            img_path = os.path.join(args.data_dir, f'{args.base}_{idx}.jpg')
            img = cv2.imread(img_path)
            if img is None:
                continue

            pts_3d = detect_face(landmarker, img)
            if pts_3d is None:
                print(f"  No face: _{idx}")
                continue

            warped = warp_perspective_smooth(img, pts_3d, cam_offset_x=offset)

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            r = infer(model, gray, 'cpu')
            gt = load_gt_angles(img_path)

            vis = warped.copy()
            scale = TARGET_H / vis.shape[0]
            vis = cv2.resize(vis, (int(vis.shape[1] * scale), TARGET_H))

            fs, th_f, lh = 0.45, 1, 20
            x0, y0 = 5, 24
            lines = [(f'Yaw:{r["yaw"]:+.1f} Pit:{r["pitch"]:+.1f}', (0, 255, 0))]
            if gt:
                lines.append((f'GT Y:{gt["yaw"]:+.1f} P:{gt["pitch"]:+.1f}', (0, 200, 255)))
            txts = [l[0] for l in lines]
            (tw, tht), _ = cv2.getTextSize(max(txts, key=len), FONT, fs, th_f)
            cv2.rectangle(vis, (x0 - 3, y0 - tht - 3), (x0 + tw + 3, y0 + lh * len(lines)), (0, 0, 0), -1)
            for i, (txt, color) in enumerate(lines):
                cv2.putText(vis, txt, (x0, y0 + i * lh), FONT, fs, color, th_f, cv2.LINE_AA)

            items_vis.append(vis)
            all_data.append({
                'cam': cam_label, 'idx': idx,
                'pred_pitch': r['pitch'], 'pred_yaw': r['yaw'],
                'gt_pitch': gt['pitch'] if gt else 0,
                'gt_yaw': gt['yaw'] if gt else 0,
            })

        if items_vis:
            row_img = np.hstack(items_vis)
            cv2.rectangle(row_img, (0, 0), (190, row_img.shape[0]), (0, 0, 0), -1)
            cv2.putText(row_img, cam_label, (6, row_img.shape[0] // 2),
                        FONT, 0.55, (255, 200, 100), 1, cv2.LINE_AA)
            all_rows.append(row_img)

    if not all_rows:
        print("ERROR: No images!")
        return

    max_w = max(r.shape[1] for r in all_rows)
    padded = []
    for r in all_rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    sep = np.ones((30, max_w, 3), dtype=np.uint8) * 50
    cv2.putText(sep, 'Smooth depth warp: A-pillar camera offset (inverse mapping)',
                (10, 20), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Pitch chart
    chart_h = 300
    chart = np.ones((chart_h, max_w, 3), dtype=np.uint8) * 25
    mx, my = 80, 40
    pw = max_w - 2 * mx
    ph = chart_h - 2 * my

    pp_all = [d['pred_pitch'] for d in all_data]
    gp_all = [d['gt_pitch'] for d in all_data]
    vmin = min(min(pp_all), min(gp_all)) - 5
    vmax = max(max(pp_all), max(gp_all)) + 5

    def v2y(v):
        return int(my + ph * (1 - (v - vmin) / (vmax - vmin)))

    for v in range(int(vmin) - 5, int(vmax) + 10, 5):
        y = v2y(v)
        if my - 5 <= y <= chart_h - my + 5:
            cv2.line(chart, (mx, y), (max_w - mx, y), (45, 45, 45), 1)
            cv2.putText(chart, f'{v:+d}' + chr(176), (5, y + 6), FONT, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    GT_COLOR = (0, 200, 255)
    color_list = [(0, 220, 80), (0, 180, 255), (80, 80, 255), (255, 100, 100)]
    PRED_COLORS = {label: color_list[i % len(color_list)] for i, (_, label) in enumerate(cam_configs)}

    gt_sub = sorted([d for d in all_data if d['cam'] == 'Frontal'], key=lambda d: d['idx'])
    pts = []
    for i, d in enumerate(gt_sub):
        x = mx + int(pw * i / max(len(gt_sub) - 1, 1))
        y = v2y(d['gt_pitch'])
        pts.append((x, y))
        cv2.circle(chart, (x, y), 4, GT_COLOR, -1)
    if len(pts) > 1:
        cv2.polylines(chart, [np.array(pts)], False, GT_COLOR, 2)

    for _, cam_label in cam_configs:
        subset = sorted([d for d in all_data if d['cam'] == cam_label], key=lambda d: d['idx'])
        pts = []
        for i, d in enumerate(subset):
            x = mx + int(pw * i / max(len(subset) - 1, 1))
            y = v2y(d['pred_pitch'])
            pts.append((x, y))
            cv2.circle(chart, (x, y), 5, PRED_COLORS[cam_label], -1)
        if len(pts) > 1:
            cv2.polylines(chart, [np.array(pts)], False, PRED_COLORS[cam_label], 2)

    cv2.putText(chart, 'Pitch vs Yaw-index (smooth depth warp A-pillar)',
                (mx, 28), FONT, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    legend = [('-- GT Pitch', GT_COLOR)]
    for _, label in cam_configs:
        legend.append((f'-- Pred: {label}', PRED_COLORS[label]))
    for i, (txt, clr) in enumerate(legend):
        cv2.putText(chart, txt, (max_w - 350, 28 + i * 22), FONT, 0.42, clr, 1, cv2.LINE_AA)

    print()
    for _, label in cam_configs:
        sub = [d for d in all_data if d['cam'] == label]
        pp = [d['pred_pitch'] for d in sub]
        print(f'{label:20s}: Pred Pitch {min(pp):+7.1f} ~ {max(pp):+7.1f}  span={max(pp)-min(pp):.1f}')
    gt_v = [d['gt_pitch'] for d in all_data if d['cam'] == 'Frontal']
    print(f'{"GT":20s}: Pitch {min(gt_v):+7.1f} ~ {max(gt_v):+7.1f}  span={max(gt_v)-min(gt_v):.1f}')

    final = np.vstack(padded + [sep, chart])
    out_path = os.path.join(args.output_dir, 'yaw_sweep_smooth_warp.png')
    cv2.imwrite(out_path, final)
    print(f'\nSaved: {out_path}  ({final.shape[1]}x{final.shape[0]})')


if __name__ == '__main__':
    main()
