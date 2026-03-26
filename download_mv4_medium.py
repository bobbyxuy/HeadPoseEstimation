import os
import sys

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import timm
import torch

print("Downloading MobileNetV4 Medium pretrained weights from mirror...")
endpoint = os.environ.get("HF_ENDPOINT")
print(f"HF_ENDPOINT: {endpoint}")

try:
    # 创建模型并下载预训练权重
    model = timm.create_model("mobilenetv4_conv_medium.e500_r256_in1k", pretrained=True)
    
    # 保存权重
    save_path = "weights/mobilenetv4_conv_medium_pretrained.pth"
    torch.save(model.state_dict(), save_path)
    
    print("Download successful!")
    print(f"Weights saved to: {save_path}")
    
    # 验证文件
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"File size: {size_mb:.2f} MB")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
