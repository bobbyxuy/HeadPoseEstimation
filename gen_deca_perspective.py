#!/usr/bin/env python3
"""
基于 DECA 的 A 柱偏置相机模拟
1. DECA 重建 3D 面部（5023 顶点，9976 三角形，UV 纹理）
2. 从偏置相机位置用透视投影重渲染
3. 推理并对比 pitch 变化
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/bobby/codes/DECA')

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from skimage.io import imread
from skimage.transform import resize


# ── DECA 初始化 ──

def load_deca():
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deca_cfg.rasterizer_type = 'soft'
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)
    return deca, device


def reconstruct(deca, img_path, device):
    """用 DECA 重建 3D 面部，返回顶点、UV纹理、faces"""
    image = imread(img_path) / 255.
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image_resized = resize(image, (224, 224), preserve_range=False)
    image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1)
    images = image_tensor.to(device)[None, ...]

    with torch.no_grad():
        codedict = deca.encode(images)
        opdict, visdict = deca.decode(codedict)

    verts = opdict['verts'][0].cpu().numpy()  # (5023, 3) 世界坐标
    uv_texture = opdict['uv_texture_gt'][0].cpu().numpy()  # (3, 256, 256) UV纹理 [0,1]
    cam = codedict['cam'][0].cpu().numpy()  # [scale, tx, ty]

    # 加载 mesh template
    from decalib.utils import util
    _, uvcoords, faces, uvfaces = util.load_obj(deca.cfg.model.topology_path)
    uvcoords = uvcoords.cpu().numpy()  # (5118, 2) [0,1]
    faces_np = faces.cpu().numpy()  # (9976, 3)
    uvfaces_np = uvfaces.cpu().numpy()  # (9976, 3)

    return {
        'verts': verts,
        'uv_texture': uv_texture,  # (3, 256, 256) RGB [0,1]
        'uvcoords': uvcoords,
        'faces': faces_np,
        'uvfaces': uvfaces_np,
        'cam': cam,  # [scale, tx, ty] orthographic
    }


# ── 透视投影渲染 ──

def render_perspective_deca(data, img_size, cam_offset_x=0.0, fov_deg=50.0):
    """
    用 DECA 重建的 mesh + UV 纹理做透视投影渲染

    cam_offset_x: 相机水平偏移（正值=相机右移）
    """
    verts = data['verts']
    uv_texture = data['uv_texture']  # (3, 256, 256)
    uvcoords = data['uvcoords']
    faces = data['faces']
    uvfaces = data['uvfaces']
    cam = data['cam']  # [scale, tx, ty] orthographic params

    # ── 将 DECA 的世界坐标转到正交投影后的坐标 ──
    # DECA 的正交投影：trans_verts = scale * verts + [tx, -ty, 0]
    scale, tx_orth, ty_orth = cam[0], cam[1], cam[2]
    trans_verts = verts * scale
    trans_verts[:, 0] += tx_orth
    trans_verts[:, 1] -= ty_orth  # DECA 翻转了 y

    # trans_verts 大致在 [-1, 1] 范围（NDC 坐标）
    # 将其转换为以面部中心为原点的 3D 坐标
    face_center = trans_verts.mean(axis=0)

    # ── 构建透视相机 ──
    fov_rad = np.radians(fov_deg)
    f = img_size / (2 * np.tan(fov_rad / 2))
    cx = img_size / 2.0
    cy = img_size / 2.0

    # 使用 DECA 的 trans_verts 作为相机坐标
    # trans_verts 已经是在正交投影下的坐标，范围约 [-1, 1]
    # 需要将其放到合理的深度

    # trans_verts 的 z 分量是深度信息
    z_range = trans_verts[:, 2].max() - trans_verts[:, 2].min()
    face_scale = trans_verts[:, 0].max() - trans_verts[:, 0].min()

    # 将 trans_verts 映射到透视相机的 3D 空间
    # 目标：面部在图像中占合理比例
    cam_pts = np.zeros_like(trans_verts)
    # x, y 直接乘以深度缩放
    # 假设面部中心在 z = f * 1.2 处
    base_z = f * 1.2
    depth_factor = base_z  # 深度缩放因子

    cam_pts[:, 0] = trans_verts[:, 0] * depth_factor
    cam_pts[:, 1] = trans_verts[:, 1] * depth_factor
    cam_pts[:, 2] = base_z + trans_verts[:, 2] * depth_factor * 0.5  # z 范围压缩

    # 应用相机偏移
    cam_pts[:, 0] -= cam_offset_x * (face_scale * depth_factor)

    # 透视投影
    z = cam_pts[:, 2]
    z = np.where(z < 1, 1, z)
    x_2d = f * cam_pts[:, 0] / z + cx
    y_2d = f * cam_pts[:, 1] / z + cy

    # ── UV 纹理渲染 ──
    output = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    zbuffer = np.full((img_size, img_size), np.inf, dtype=np.float64)

    # UV 纹理图 (3, H_tex, W_tex)
    tex_h, tex_w = uv_texture.shape[1], uv_texture.shape[2]
    tex_img = (uv_texture.transpose(1, 2, 0) * 255).astype(np.uint8)  # (H_tex, W_tex, 3)

    # 按深度排序（远到近）
    tri_z = np.mean(z[faces], axis=1)
    order = np.argsort(-tri_z)

    for ti in order:
        idx = faces[ti]
        uv_idx = uvfaces[ti]

        # 目标坐标
        dst = np.array([
            [x_2d[idx[0]], y_2d[idx[0]]],
            [x_2d[idx[1]], y_2d[idx[1]]],
            [x_2d[idx[2]], y_2d[idx[2]]],
        ], dtype=np.float64)

        # 跳过画面外
        if (dst[:, 0].max() < 0 or dst[:, 0].min() >= img_size or
            dst[:, 1].max() < 0 or dst[:, 1].min() >= img_size):
            continue

        # 跳过退化三角形
        e1 = dst[1] - dst[0]
        e2 = dst[2] - dst[0]
        area = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(area) < 0.5:
            continue

        # UV 纹理坐标
        uv_dst = np.array([
            [uvcoords[uv_idx[0], 0] * tex_w, uvcoords[uv_idx[0], 1] * tex_h],
            [uvcoords[uv_idx[1], 0] * tex_w, uvcoords[uv_idx[1], 1] * tex_h],
            [uvcoords[uv_idx[2], 0] * tex_w, uvcoords[uv_idx[2], 1] * tex_h],
        ], dtype=np.float64)

        # 仿射变换：从屏幕坐标到 UV 坐标
        # uv = A * screen + b
        # [uv0 uv1 uv2] = [a b c; d e f] * [dst0 dst1 dst2; 1 1 1]
        S = np.array([
            [dst[0, 0], dst[1, 0], dst[2, 0]],
            [dst[0, 1], dst[1, 1], dst[2, 1]],
            [1, 1, 1],
        ])
        U = np.array([
            [uv_dst[0, 0], uv_dst[1, 0], uv_dst[2, 0]],
            [uv_dst[0, 1], uv_dst[1, 1], uv_dst[2, 1]],
        ])

        try:
            A_uv = U @ np.linalg.inv(S)  # (2, 3)
        except np.linalg.LinAlgError:
            continue

        # 光栅化
        x_min = max(0, int(np.floor(dst[:, 0].min())))
        x_max = min(img_size - 1, int(np.ceil(dst[:, 0].max())))
        y_min = max(0, int(np.floor(dst[:, 1].min())))
        y_max = min(img_size - 1, int(np.ceil(dst[:, 1].max())))

        inv_area = 1.0 / area
        v0 = dst[0]

        for py in range(y_min, y_max + 1):
            px_arr = np.arange(x_min, x_max + 1, dtype=np.float64)
            rx = px_arr - v0[0]
            ry = float(py) - v0[1]

            w1 = (rx * e2[1] - ry * e2[0]) * inv_area
            w2 = (ry * e1[0] - rx * e1[1]) * inv_area
            w3 = 1.0 - w1 - w2

            valid = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)

            for j in np.where(valid)[0]:
                px = x_min + j
                pz = w1[j] * z[idx[0]] + w2[j] * z[idx[1]] + w3[j] * z[idx[2]]
                if pz < zbuffer[py, px]:
                    zbuffer[py, px] = pz

                    # 计算 UV 纹理坐标
                    u_tex = A_uv[0, 0] * px_arr[j] + A_uv[0, 1] * py + A_uv[0, 2]
                    v_tex = A_uv[1, 0] * px_arr[j] + A_uv[1, 1] * py + A_uv[1, 2]

                    # 双线性采样
                    ix = int(np.clip(np.floor(u_tex), 0, tex_w - 2))
                    iy = int(np.clip(np.floor(v_tex), 0, tex_h - 2))
                    fx = np.clip(u_tex - ix, 0, 1)
                    fy = np.clip(v_tex - iy, 0, 1)

                    c = (tex_img[iy, ix] * (1 - fx) * (1 - fy) +
                         tex_img[iy, ix + 1] * fx * (1 - fy) +
                         tex_img[iy + 1, ix] * (1 - fx) * fy +
                         tex_img[iy + 1, ix + 1] * fx * fy)
                    output[py, px] = np.clip(c, 0, 255).astype(np.uint8)

    return output


# ── 主程序 ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description='DECA 3D Face A-pillar simulation')
    parser.add_argument('--data_dir', default='data/300W_LP/AFW')
    parser.add_argument('--base', default='AFW_1130084326_3')
    parser.add_argument('--output_dir', default='output_yaw_sweep')
    parser.add_argument('--img_size', type=int, default=450)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading DECA model...")
    deca, device = load_deca()
    print(f"  Device: {device}")

    print("Loading head pose model...")
    from infer_headpose import load_model, infer, load_gt_angles
    model = load_model('weights/mobilenetv4_medium_pretrained/best_checkpoint.ckpt',
                        'mobilenetv4_conv_medium', 'cpu', from_ckpt=True)

    cam_configs = [
        (0.0,  'Frontal'),
        (0.3,  'A-pillar mild'),
        (0.6,  'A-pillar medium'),
        (1.0,  'A-pillar strong'),
    ]

    TARGET_H = 220
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    all_rows = []
    all_data = []

    for offset, cam_label in cam_configs:
        items_vis = []
        for idx in range(18):
            img_path = os.path.join(args.data_dir, f'{args.base}_{idx}.jpg')
            if not os.path.exists(img_path):
                continue

            print(f"  Processing {cam_label} _{idx}...")
            try:
                data = reconstruct(deca, img_path, device)
            except Exception as e:
                print(f"    Error: {e}")
                continue

            rendered = render_perspective_deca(
                data, args.img_size, cam_offset_x=offset)

            gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY) if rendered.ndim == 3 else rendered
            r = infer(model, gray, 'cpu')
            gt = load_gt_angles(img_path)

            vis = rendered.copy()
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
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
            cv2.rectangle(row_img, (0, 0), (200, row_img.shape[0]), (0, 0, 0), -1)
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
    cv2.putText(sep, 'DECA 3D Face Re-rendered: A-pillar camera offset simulation',
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

    cv2.putText(chart, 'Pitch vs Yaw-index (DECA 3D re-rendered, A-pillar)',
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
    out_path = os.path.join(args.output_dir, 'yaw_sweep_deca.png')
    cv2.imwrite(out_path, final)
    print(f'\nSaved: {out_path}  ({final.shape[1]}x{final.shape[0]})')


if __name__ == '__main__':
    main()
