"""
Test official WHENet pretrained model (WHENet.h5) on AFLW2000.
Requires TensorFlow 2.x + Keras + efficientnet package.
"""

import os
import sys
import numpy as np
from PIL import Image
from scipy import io
import tensorflow as tf
from tensorflow import keras

# ── Load official WHENet model ───────────────────────────────────────────────

def load_official_whenet(h5_path):
    """Load the official WHENet.h5 Keras model."""
    # The official model uses efficientnet package (not tf.keras.applications)
    try:
        import efficientnet.keras as efn
    except ImportError:
        print("ERROR: 'efficientnet' package not found. Install via: pip install efficientnet")
        sys.exit(1)
    
    # Rebuild model architecture (same as official whenet.py)
    base_model = efn.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights=None)
    out = base_model.output
    out = keras.layers.GlobalAveragePooling2D()(out)
    fc_yaw   = keras.layers.Dense(name='yaw_new',   units=120)(out)
    fc_pitch = keras.layers.Dense(name='pitch_new', units=66)(out)
    fc_roll  = keras.layers.Dense(name='roll_new',  units=66)(out)
    model = keras.models.Model(inputs=base_model.input, outputs=[fc_yaw, fc_pitch, fc_roll])
    
    # Load weights
    model.load_weights(h5_path)
    print(f"✓ Loaded official WHENet from {h5_path}")
    return model


# ── Prediction helpers ───────────────────────────────────────────────────────

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def predict_angles(model, img_batch):
    """
    img_batch: (B, 224, 224, 3), float32, range [0,1], already normalized.
    Returns: yaw, pitch, roll in degrees (B,)
    """
    yaw_logit, pitch_logit, roll_logit = model.predict(img_batch, verbose=0)
    
    idx_yaw   = np.arange(120, dtype=np.float32)
    idx_pitch = np.arange(66,  dtype=np.float32)
    idx_roll  = np.arange(66,  dtype=np.float32)
    
    yaw_prob   = softmax(yaw_logit)
    pitch_prob = softmax(pitch_logit)
    roll_prob  = softmax(roll_logit)
    
    yaw   = np.sum(yaw_prob   * idx_yaw,   axis=1) * 3.0 - 180.0
    pitch = np.sum(pitch_prob * idx_pitch, axis=1) * 3.0 -  99.0
    roll  = np.sum(roll_prob  * idx_roll,  axis=1) * 3.0 -  99.0
    
    return yaw, pitch, roll


# ── AFLW2000 dataset ─────────────────────────────────────────────────────────

def load_aflw2000(root_dir, max_angle=99.0):
    """Scan AFLW2000 directory, return list of (img_path, yaw_gt, pitch_gt, roll_gt)."""
    samples = []
    for fname in os.listdir(root_dir):
        if not fname.endswith('.jpg'):
            continue
        stem = fname[:-4]
        img_path = os.path.join(root_dir, fname)
        mat_path = os.path.join(root_dir, stem + '.mat')
        if not os.path.exists(mat_path):
            continue
        
        lbl = io.loadmat(mat_path)
        pitch = float(lbl['Pose_Para'][0][0]) * 180.0 / np.pi
        yaw   = float(lbl['Pose_Para'][0][1]) * 180.0 / np.pi
        roll  = float(lbl['Pose_Para'][0][2]) * 180.0 / np.pi
        
        # Filter same as training (pitch/roll ≤99°, yaw ≤180°)
        if abs(pitch) <= max_angle and abs(yaw) <= 180.0 and abs(roll) <= max_angle:
            samples.append((img_path, mat_path, yaw, pitch, roll))
    
    print(f"Loaded {len(samples)} samples from AFLW2000")
    return samples


def preprocess_image(img_path, mat_path):
    """
    Crop face region (same as official demo.py), resize to 224x224, normalize.
    Returns: (224, 224, 3) float32 array, range [0,1], normalized.
    """
    img = Image.open(img_path).convert('RGB')
    lbl = io.loadmat(mat_path)
    pt2d = lbl['pt2d']
    
    x_min = np.min(pt2d[0])
    x_max = np.max(pt2d[0])
    y_min = np.min(pt2d[1])
    y_max = np.max(pt2d[1])
    
    # Expand bbox slightly (same as eval dataset)
    x_min -= 0.2 * (x_max - x_min)
    x_max += 0.2 * (x_max - x_min)
    y_min -= 0.2 * (y_max - y_min)
    y_max += 0.2 * (y_max - y_min)
    
    img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    img = img.resize((224, 224), Image.BILINEAR)
    
    # Convert to array and normalize (same as official whenet.py)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std
    
    return img_arr


# ── Evaluation ───────────────────────────────────────────────────────────────

def angle_mae(pred, gt):
    """MAE with wraparound correction."""
    diff = np.abs(pred - gt)
    diff = np.minimum(diff, 360.0 - diff)
    return diff


def evaluate_aflw2000(model, aflw_root, batch_size=32):
    samples = load_aflw2000(aflw_root)
    
    yaw_err = pitch_err = roll_err = 0.0
    total = len(samples)
    
    for i in range(0, total, batch_size):
        batch = samples[i:i+batch_size]
        imgs = []
        yaw_gts = []
        pitch_gts = []
        roll_gts = []
        
        for img_path, mat_path, yaw_gt, pitch_gt, roll_gt in batch:
            img_arr = preprocess_image(img_path, mat_path)
            imgs.append(img_arr)
            yaw_gts.append(yaw_gt)
            pitch_gts.append(pitch_gt)
            roll_gts.append(roll_gt)
        
        imgs = np.stack(imgs, axis=0)  # (B, 224, 224, 3)
        yaw_pred, pitch_pred, roll_pred = predict_angles(model, imgs)
        
        yaw_gts   = np.array(yaw_gts)
        pitch_gts = np.array(pitch_gts)
        roll_gts  = np.array(roll_gts)
        
        yaw_err   += np.sum(angle_mae(yaw_pred,   yaw_gts))
        pitch_err += np.sum(angle_mae(pitch_pred, pitch_gts))
        roll_err  += np.sum(angle_mae(roll_pred,  roll_gts))
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {i+len(batch)}/{total} samples...")
    
    yaw_mae   = yaw_err   / total
    pitch_mae = pitch_err / total
    roll_mae  = roll_err  / total
    mae       = (yaw_mae + pitch_mae + roll_mae) / 3.0
    
    print("\n" + "="*60)
    print("Official WHENet evaluation on AFLW2000:")
    print(f"  Yaw:   {yaw_mae:.4f}°")
    print(f"  Pitch: {pitch_mae:.4f}°")
    print(f"  Roll:  {roll_mae:.4f}°")
    print(f"  MAE:   {mae:.4f}°")
    print("="*60)
    
    return mae


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--h5', type=str, default='WHENet_official.h5', help='Path to WHENet.h5')
    p.add_argument('--aflw', type=str, default='data/AFLW2000', help='AFLW2000 root dir')
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    
    model = load_official_whenet(args.h5)
    evaluate_aflw2000(model, args.aflw, batch_size=args.batch_size)
