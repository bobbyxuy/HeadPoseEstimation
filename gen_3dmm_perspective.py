#!/usr/bin/env python3
"""
基于 3DMM 的 A 柱偏置相机模拟
从 .mat 文件读取 3DMM 参数，改变相机位置重新渲染，模拟 A 柱视角

原理：
  1. 读取 Shape_Para + Exp_Para → 3D 顶点
  2. 读取 Tex_Para → 纹理颜色
  3. 读取 Pose_Para → 原始旋转/平移
  4. 改变相机位置（平移到侧面）→ 重新投影渲染
"""
import sys, os
import argparse
import cv2
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 3DMM 基向量加载 ──

def load_3dmm_bases(code_dir='data/AFLW2000/Code'):
    """加载 3DMM 基向量（形状、表情、纹理、三角形）"""
    shape_path = os.path.join(code_dir, 'Model_Shape_Sim.mat')
    exp_path = os.path.join(code_dir, 'Model_Exp.mat')

    if not os.path.exists(shape_path):
        # 尝试 300W_LP 目录
        shape_path = os.path.join('data/300W_LP/Code', 'Model_Shape_Sim.mat')
        exp_path = os.path.join('data/300W_LP/Code', 'Model_Exp.mat')

    shape_data = sio.loadmat(shape_path)
    exp_data = sio.loadmat(exp_path)

    # 形状基: w (3N x 199), 平均形状: mu_shape (3N x 1)
    w = shape_data.get('w', shape_data.get('shape_basis', None))
    mu_shape = shape_data.get('mu_shape', shape_data.get('mu', None))

    # 表情基: w_exp (3N x 29), 平均表情: mu_exp (3N x 1)
    w_exp = exp_data.get('w_exp', exp_data.get('exp_basis', None))
    mu_exp = exp_data.get('mu_exp', exp_data.get('mu', None))

    # 纹理基
    mu_tex = shape_data.get('mu_tex', None)
    w_tex = shape_data.get('w_tex', None)

    # 三角形
    tri = shape_data.get('tri', shape_data.get('triangles', None))

    print(f"Shape: w={w.shape if w is not None else None}, mu={mu_shape.shape if mu_shape is not None else None}")
    print(f"Exp:   w_exp={w_exp.shape if w_exp is not None else None}, mu_exp={mu_exp.shape if mu_exp is not None else None}")
    print(f"Tex:   mu_tex={mu_tex.shape if mu_tex is not None else None}, w_tex={w_tex.shape if w_tex is not None else None}")
    print(f"Tri:   {tri.shape if tri is not None else None}")

    return {
        'w': w, 'mu_shape': mu_shape,
        'w_exp': w_exp, 'mu_exp': mu_exp,
        'mu_tex': mu_tex, 'w_tex': w_tex,
        'tri': tri,
    }


def reconstruct_3d_face(mat_data, bases):
    """从 .mat 参数重建 3D 顶点和纹理"""
    shape_para = mat_data['Shape_Para']  # (199, 1)
    exp_para = mat_data['Exp_Para']      # (29, 1)
    tex_para = mat_data['Tex_Para']      # (199, 1)

    # 顶点 = 平均形状 + 形状基 x 形状参数 + 平均表情 + 表情基 x 表情参数
    mu = bases['mu_shape'] + bases['mu_exp']
    vertex = mu + bases['w'] @ shape_para + bases['w_exp'] @ exp_para
    vertex = vertex.reshape(3, -1)  # (3, N)

    # 纹理 = 平均纹理 + 纹理基 x 纹理参数
    tex = None
    if bases['mu_tex'] is not None and bases['w_tex'] is not None:
        tex = bases['mu_tex'] + bases['w_tex'] @ tex_para
        tex = tex.reshape(3, -1)  # (3, N)

    return vertex, tex


# ── 相机模型 ──

def rotation_matrix_euler(pitch, yaw, roll):
    """从 Euler 角（弧度）构建旋转矩阵"""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)],
    ])
    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0, 0, 1],
    ])
    return Rz @ Ry @ Rx


def project_perspective(vertex, R, t, f, cx, cy, h, w):
    """透视投影：3D → 2D
    vertex: (3, N)
    R: (3, 3) 旋转矩阵
    t: (3,) 平移
    f: 焦距
    cx, cy: 主点
    """
    # 世界坐标 → 相机坐标
    cam_pts = R @ vertex + t.reshape(3, 1)

    # 透视除法
    z = cam_pts[2:3, :]  # (1, N)
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)

    x_2d = f * cam_pts[0:1, :] / z + cx
    y_2d = f * cam_pts[1:2, :] / z + cy

    return x_2d.flatten(), y_2d.flatten(), z.flatten(), cam_pts


def render_mesh(vertex_2d_x, vertex_2d_y, vertex_z, tex, tri, h, w, img_bg=None):
    """简单 Z-buffer 三角形光栅化渲染"""
    if img_bg is not None:
        color_img = img_bg.copy()
    else:
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
    zbuffer = np.full((h, w), np.inf, dtype=np.float64)

    # 按三角形平均深度排序（远到近）
    tri_z = np.mean(vertex_z[tri], axis=1)
    order = np.argsort(-tri_z)  # 远的先画

    for ti in order:
        idx = tri[ti]
        pts = np.array([
            [vertex_2d_x[idx[0]], vertex_2d_y[idx[0]]],
            [vertex_2d_x[idx[1]], vertex_2d_y[idx[1]]],
            [vertex_2d_x[idx[2]], vertex_2d_y[idx[2]]],
        ], dtype=np.float32)

        # 跳过画面外的三角形
        if (pts[:, 0].max() < 0 or pts[:, 0].min() >= w or
            pts[:, 1].max() < 0 or pts[:, 1].min() >= h):
            continue

        # 纹理颜色取平均
        if tex is not None:
            c = tex[:, idx].mean(axis=1)
            color = (int(np.clip(c[0], 0, 255)),
                     int(np.clip(c[1], 0, 255)),
                     int(np.clip(c[2], 0, 255)))
        else:
            color = (180, 180, 180)

        cv2.fillConvexPoly(color_img, pts.astype(np.int32), color)

    return color_img


def render_with_appearance(vertex, tex, mat_data, bases, cam_offset_x=0, cam_offset_y=0, img_size=450):
    """
    完整渲染管线：3DMM → 3D → 改变相机位置 → 2D 渲染

    cam_offset_x: 相机水平偏移（模拟 A 柱偏置），正值=相机右移
    """
    pose = mat_data['Pose_Para'][0]
    pitch, yaw, roll = pose[0], pose[1], pose[2]
    tx, ty, tz = pose[3], pose[4], pose[5]
    f = pose[6]

    h, w = img_size, img_size
    cx, cy = w / 2.0, h / 2.0

    # 头部旋转矩阵
    R_head = rotation_matrix_euler(pitch, yaw, roll)

    # 原始相机位置下的投影参数
    # 原始：t = [tx, ty, tz], f = scale
    # 偏置相机：改变 t 中的 x 分量（相机在侧面）
    # 相机右移相当于所有点向左平移

    # 等效做法：不改相机，而是把 3D 点在相机坐标下做额外平移
    # 先做原始变换到相机坐标
    cam_pts = R_head @ vertex + np.array([[tx], [ty], [tz]])

    # 应用相机偏移：相机右移 → 场景中所有点左移
    # 同时调整 tz 使面部仍在合理距离
    cam_pts[0, :] -= cam_offset_x * f  # 水平偏移

    # 透视投影
    z = cam_pts[2:3, :]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)

    x_2d = f * cam_pts[0:1, :] / z + cx
    y_2d = f * cam_pts[1:2, :] / z + cy

    # 处理光照（简化版）
    color_per_tri = None
    if tex is not None:
        illum_para = mat_data.get('Illum_Para', np.zeros((1, 10)))[0]
        color_para = mat_data.get('Color_Para', np.zeros((1, 7)))[0]
        tri = bases['tri']

        # 法向量
        v0 = vertex[:, tri[:, 0]]
        v1 = vertex[:, tri[:, 1]]
        v2 = vertex[:, tri[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=0, keepdims=True)
        norms = np.where(norms < 1e-10, 1, norms)
        normals = normals / norms

        # 简单方向光
        light_dir = np.array([0, 0, 1])  # 前方光源
        diffuse = np.clip(normals.T @ light_dir, 0, 1)
        ambient = 0.4
        lighting = ambient + (1 - ambient) * diffuse

        # 每个三角形的颜色
        tri_tex = np.stack([
            tex[:, tri[:, i]].mean(axis=1) for i in range(3)
        ], axis=0).mean(axis=0)  # (3, n_tri)

        color_per_tri = tri_tex * lighting[np.newaxis, :]

    # 渲染
    img = np.zeros((h, w, 3), dtype=np.uint8)
    zbuffer = np.full((h, w), np.inf)

    tri = bases['tri']
    tri_z = np.mean(cam_pts[2, tri], axis=1)
    order = np.argsort(-tri_z)

    for ti in order:
        idx = tri[ti]
        pts = np.array([
            [x_2d[0, idx[0]], y_2d[0, idx[0]]],
            [x_2d[0, idx[1]], y_2d[0, idx[1]]],
            [x_2d[0, idx[2]], y_2d[0, idx[2]]],
        ], dtype=np.float32)

        if (pts[:, 0].max() < 0 or pts[:, 0].min() >= w or
            pts[:, 1].max() < 0 or pts[:, 1].min() >= h):
            continue

        if color_per_tri is not None:
            c = color_per_tri[:, ti]
            color = (int(np.clip(c[2], 0, 255)),
                     int(np.clip(c[1], 0, 255)),
                     int(np.clip(c[0], 0, 255)))
        else:
            color = (180, 180, 180)

        cv2.fillConvexPoly(img, pts.astype(np.int32), color)

    return img


# ── 主程序 ──

def main():
    parser = argparse.ArgumentParser(description='3DMM based A-pillar camera simulation')
    parser.add_argument('--base', default='AFW_1130084326_3', help='Image base name')
    parser.add_argument('--data_dir', default='data/300W_LP/AFW', help='Image directory')
    parser.add_argument('--code_dir', default='data/300W_LP/Code', help='3DMM model directory')
    parser.add_argument('--output_dir', default='output_yaw_sweep', help='Output directory')
    parser.add_argument('--offset', type=float, default=0.5, help='Camera offset (meters)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading 3DMM bases...")
    bases = load_3dmm_bases(args.code_dir)

    print("Loading model for inference...")
    from infer_headpose import load_model, infer, load_gt_angles
    model = load_model('weights/mobilenetv4_medium_pretrained/best_checkpoint.ckpt',
                        'mobilenetv4_conv_medium', 'cpu', from_ckpt=True)

    # 三个相机配置
    cam_configs = [
        (0.0,  'Frontal'),
        (0.3,  'A-pillar 30cm'),
        (0.6,  'A-pillar 60cm'),
    ]

    target_h = 220
    font = cv2.FONT_HERSHEY_SIMPLEX
    all_rows = []
    all_data = []

    for offset, cam_label in cam_configs:
        items_vis = []
        for idx in range(18):
            mat_path = os.path.join(args.data_dir, f'{args.base}_{idx}.mat')
            if not os.path.exists(mat_path):
                continue

            mat_data = sio.loadmat(mat_path)
            vertex, tex = reconstruct_3d_face(mat_data, bases)
            img_size = 450

            # 渲染 3DMM 视图
            rendered = render_with_appearance(vertex, tex, mat_data, bases,
                                               cam_offset_x=offset, img_size=img_size)

            # 推理
            gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY) if len(rendered.shape) == 3 else rendered
            r = infer(model, gray, 'cpu')
            gt = None
            gt_path = mat_path
            gt = load_gt_angles(gt_path.replace('.mat', '.jpg'))

            # 可视化
            vis = rendered.copy()
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            scale = target_h / vis.shape[0]
            vis = cv2.resize(vis, (int(vis.shape[1] * scale), target_h))

            fs, th_f, lh = 0.45, 1, 20
            x0, y0 = 5, 24
            lines = [
                (f'Yaw:{r["yaw"]:+.1f} Pit:{r["pitch"]:+.1f}', (0, 255, 0)),
            ]
            if gt:
                lines.append((f'GT Y:{gt["yaw"]:+.1f} P:{gt["pitch"]:+.1f}', (0, 200, 255)))
            txts = [l[0] for l in lines]
            (tw, tht), _ = cv2.getTextSize(max(txts, key=len), font, fs, th_f)
            cv2.rectangle(vis, (x0 - 3, y0 - tht - 3), (x0 + tw + 3, y0 + lh * len(lines)), (0, 0, 0), -1)
            for i, (txt, color) in enumerate(lines):
                cv2.putText(vis, txt, (x0, y0 + i * lh), font, fs, color, th_f, cv2.LINE_AA)

            items_vis.append(vis)
            all_data.append({
                'cam': cam_label, 'idx': idx,
                'pred_pitch': r['pitch'], 'pred_yaw': r['yaw'],
                'gt_pitch': gt['pitch'] if gt else 0,
                'gt_yaw': gt['yaw'] if gt else 0,
            })

        if items_vis:
            row_img = np.hstack(items_vis)
            cv2.rectangle(row_img, (0, 0), (180, row_img.shape[0]), (0, 0, 0), -1)
            cv2.putText(row_img, cam_label, (6, row_img.shape[0] // 2),
                        font, 0.55, (255, 200, 100), 1, cv2.LINE_AA)
            all_rows.append(row_img)

    if not all_rows:
        print("ERROR: No images generated!")
        return

    # 统一宽度
    max_w = max(r.shape[1] for r in all_rows)
    padded = []
    for r in all_rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    # 分隔
    sep = np.ones((30, max_w, 3), dtype=np.uint8) * 50
    cv2.putText(sep, '3DMM Re-rendered: yaw sweep with A-pillar camera offset',
                (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Pitch 曲线 ──
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

    # 网格
    for v in range(int(vmin) - 5, int(vmax) + 10, 5):
        y = v2y(v)
        if my - 5 <= y <= chart_h - my + 5:
            cv2.line(chart, (mx, y), (max_w - mx, y), (45, 45, 45), 1)
            cv2.putText(chart, f'{v:+d}' + chr(176), (5, y + 6), font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    GT_COLOR = (0, 200, 255)
    PRED_COLORS = {
        'Frontal':        (0, 220, 80),
        'A-pillar 30cm':  (0, 180, 255),
        'A-pillar 60cm':  (80, 80, 255),
    }

    # GT
    gt_sub = sorted([d for d in all_data if d['cam'] == 'Frontal'], key=lambda d: d['idx'])
    pts = []
    for i, d in enumerate(gt_sub):
        x = mx + int(pw * i / max(len(gt_sub) - 1, 1))
        y = v2y(d['gt_pitch'])
        pts.append((x, y))
        cv2.circle(chart, (x, y), 4, GT_COLOR, -1)
    if len(pts) > 1:
        cv2.polylines(chart, [np.array(pts)], False, GT_COLOR, 2)

    # Pred per camera
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

    # 图例
    cv2.putText(chart, 'Pitch vs Yaw (3DMM re-rendered with camera offset)',
                (mx, 28), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    legend = [
        ('-- GT Pitch', GT_COLOR),
        ('-- Pred: Frontal', PRED_COLORS['Frontal']),
        ('-- Pred: A-pillar 30cm', PRED_COLORS['A-pillar 30cm']),
        ('-- Pred: A-pillar 60cm', PRED_COLORS['A-pillar 60cm']),
    ]
    for i, (txt, clr) in enumerate(legend):
        cv2.putText(chart, txt, (max_w - 350, 28 + i * 22), font, 0.42, clr, 1, cv2.LINE_AA)

    # 统计
    print()
    for _, cam_label in cam_configs:
        sub = [d for d in all_data if d['cam'] == cam_label]
        pp = [d['pred_pitch'] for d in sub]
        print(f'{cam_label:18s}: Pred Pitch {min(pp):+7.1f} ~ {max(pp):+7.1f}  span={max(pp)-min(pp):.1f}')
    gt_v = [d['gt_pitch'] for d in all_data if d['cam'] == 'Frontal']
    print(f'{"GT":18s}: Pitch {min(gt_v):+7.1f} ~ {max(gt_v):+7.1f}  span={max(gt_v)-min(gt_v):.1f}')

    final = np.vstack(padded + [sep, chart])
    out_path = os.path.join(args.output_dir, 'yaw_sweep_3dmm_perspective.png')
    cv2.imwrite(out_path, final)
    print(f'\nSaved: {out_path}  ({final.shape[1]}x{final.shape[0]})')


if __name__ == '__main__':
    main()
