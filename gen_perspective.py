#!/usr/bin/env python3
"""生成模拟 A 柱偏置相机视角的透视变换图"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_headpose import load_model, infer, load_gt_angles
import cv2
import numpy as np

model = load_model('weights/mobilenetv4_medium_pretrained/best_checkpoint.ckpt',
                    'mobilenetv4_conv_medium', 'cpu', from_ckpt=True)

BASE = 'AFW_1130084326_3'
FRONTAL_DIR = 'data/300W_LP/AFW'
OUTDIR = 'output_yaw_sweep'
os.makedirs(OUTDIR, exist_ok=True)


def apply_perspective(img, offset_px=0):
    """模拟侧面偏置相机（A 柱视角）
    offset_px: 相机水平偏移像素，越大偏得越远
    效果：左侧压缩（远离相机），右侧拉伸（靠近相机），近大远小
    """
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # 相机在右侧 -> 右边近（大）左边远（小）
    # 顶部梯形收窄，底部向右扩展
    dst = np.float32([
        [offset_px * 0.4,           offset_px * 0.15],    # 左上：右移+下移
        [w - offset_px * 0.1,      -offset_px * 0.1],     # 右上：微收
        [w + offset_px * 0.6,      h + offset_px * 0.1],  # 右下：大幅外扩
        [-offset_px * 0.3,         h - offset_px * 0.05], # 左下：内收
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    # 画布加大以容纳变形
    new_w = w + abs(offset_px)
    result = cv2.warpPerspective(img, M, (new_w, h), borderValue=0)
    return result


CAM_CONFIGS = [
    (0,   'Frontal'),
    (60,  'A-pillar mild'),
    (90,  'A-pillar strong'),
]

TARGET_H = 180
FONT = cv2.FONT_HERSHEY_SIMPLEX
all_rows_data = []
all_pitch_data = []  # list of dicts

for offset, cam_label in CAM_CONFIGS:
    items_vis = []
    for idx in range(18):
        img_path = os.path.join(FRONTAL_DIR, f'{BASE}_{idx}.jpg')
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        warped = apply_perspective(gray, offset)
        r = infer(model, warped, 'cpu')
        gt = load_gt_angles(img_path)

        vis = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        scale = TARGET_H / vis.shape[0]
        vis = cv2.resize(vis, (int(vis.shape[1] * scale), TARGET_H))

        fs, th_f, lh = 0.4, 1, 18
        x0, y0 = 4, 22
        lines = [(f'Pit:{r["pitch"]:+.1f}', (0, 255, 0))]
        if gt:
            lines.append((f'GT:{gt["pitch"]:+.1f}', (0, 200, 255)))
        txts = [l[0] for l in lines]
        (tw, tht), _ = cv2.getTextSize(max(txts, key=len), FONT, fs, th_f)
        cv2.rectangle(vis, (x0 - 2, y0 - tht - 2), (x0 + tw + 2, y0 + lh + 2), (0, 0, 0), -1)
        for i, (txt, color) in enumerate(lines):
            cv2.putText(vis, txt, (x0, y0 + i * lh), FONT, fs, color, th_f, cv2.LINE_AA)

        items_vis.append(vis)
        all_pitch_data.append({
            'cam': cam_label, 'idx': idx,
            'pred_pitch': r['pitch'],
            'gt_pitch': gt['pitch'] if gt else 0,
        })

    row_img = np.hstack(items_vis)
    cv2.rectangle(row_img, (0, 0), (160, row_img.shape[0]), (0, 0, 0), -1)
    cv2.putText(row_img, cam_label, (6, row_img.shape[0] // 2),
                FONT, 0.5, (255, 200, 100), 1, cv2.LINE_AA)
    all_rows_data.append(row_img)

# 统一宽度
max_w = max(r.shape[1] for r in all_rows_data)
padded = []
for r in all_rows_data:
    if r.shape[1] < max_w:
        pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
        r = np.hstack([r, pad])
    padded.append(r)

# 分隔
sep = np.ones((25, max_w, 3), dtype=np.uint8) * 60
cv2.putText(sep, 'Yaw -3 ~ -88 | Same person, different camera offsets (simulating A-pillar)',
            (10, 17), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

# ── Pitch 曲线 ──
chart_h = 280
chart = np.ones((chart_h, max_w, 3), dtype=np.uint8) * 25
mx, my = 80, 35
pw = max_w - 2 * mx
ph = chart_h - 2 * my

pp_all = [d['pred_pitch'] for d in all_pitch_data]
gp_all = [d['gt_pitch'] for d in all_pitch_data]
vmin = min(min(pp_all), min(gp_all)) - 5
vmax = max(max(pp_all), max(gp_all)) + 5


def v2y(v):
    return int(my + ph * (1 - (v - vmin) / (vmax - vmin)))


# 网格
for v in range(int(vmin) - 5, int(vmax) + 10, 5):
    y = v2y(v)
    if my - 5 <= y <= chart_h - my + 5:
        cv2.line(chart, (mx, y), (max_w - mx, y), (45, 45, 45), 1)
        cv2.putText(chart, f'{v:+d}' + chr(176), (5, y + 6), FONT, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

COLORS_GT = (0, 200, 255)
COLORS_PRED = {
    'Frontal':          (0, 220, 80),
    'A-pillar mild':    (0, 180, 255),
    'A-pillar strong':  (80, 80, 255),
}

# GT 只画一次
gt_sub = sorted([d for d in all_pitch_data if d['cam'] == 'Frontal'], key=lambda d: d['idx'])
pts_gt = []
for i, d in enumerate(gt_sub):
    x = mx + int(pw * i / max(len(gt_sub) - 1, 1))
    y = v2y(d['gt_pitch'])
    pts_gt.append((x, y))
    cv2.circle(chart, (x, y), 4, COLORS_GT, -1)
if len(pts_gt) > 1:
    cv2.polylines(chart, [np.array(pts_gt)], False, COLORS_GT, 2)

# Pred 每个 cam
for _, cam_label in CAM_CONFIGS:
    subset = sorted([d for d in all_pitch_data if d['cam'] == cam_label], key=lambda d: d['idx'])
    pts = []
    for i, d in enumerate(subset):
        x = mx + int(pw * i / max(len(subset) - 1, 1))
        y = v2y(d['pred_pitch'])
        pts.append((x, y))
        cv2.circle(chart, (x, y), 5, COLORS_PRED[cam_label], -1)
    if len(pts) > 1:
        cv2.polylines(chart, [np.array(pts)], False, COLORS_PRED[cam_label], 2)

# 图例
cv2.putText(chart, 'Pitch vs Yaw (simulated A-pillar perspective)',
            (mx, 24), FONT, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
legend = [
    ('-- GT Pitch', COLORS_GT),
    ('-- Pred: Frontal', COLORS_PRED['Frontal']),
    ('-- Pred: A-pillar mild', COLORS_PRED['A-pillar mild']),
    ('-- Pred: A-pillar strong', COLORS_PRED['A-pillar strong']),
]
for i, (txt, clr) in enumerate(legend):
    cv2.putText(chart, txt, (max_w - 330, 24 + i * 22), FONT, 0.4, clr, 1, cv2.LINE_AA)

# 统计
for _, cam_label in CAM_CONFIGS:
    sub = [d['pred_pitch'] for d in all_pitch_data if d['cam'] == cam_label]
    print(f'{cam_label:18s}: Pred Pitch {min(sub):+7.1f} ~ {max(sub):+7.1f}  span={max(sub)-min(sub):.1f}')
gt_v = [d['gt_pitch'] for d in all_pitch_data if d['cam'] == 'Frontal']
print(f'{"GT":18s}: Pitch {min(gt_v):+7.1f} ~ {max(gt_v):+7.1f}  span={max(gt_v)-min(gt_v):.1f}')

final = np.vstack(padded + [sep, chart])
out_path = os.path.join(OUTDIR, 'yaw_sweep_perspective.png')
cv2.imwrite(out_path, final)
print(f'\nSaved: {out_path}  ({final.shape[1]}x{final.shape[0]})')
