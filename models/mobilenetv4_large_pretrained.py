import torch
import torch.nn as nn
import timm
from utils.general import compute_rotation_matrix_from_ortho6d
import os

class MobileNetV4LargePretrained(nn.Module):
    """MobileNetV4 Conv Large with ImageNet pretrained weights"""
    def __init__(self, pretrained=True, num_classes=6):
        super().__init__()
        
        # Create backbone without pretrained (load manually from local file)
        self.backbone = timm.create_model(
            "mobilenetv4_conv_large", 
            pretrained=False,  # Don't download from HF
            num_classes=0, 
            global_pool="avg"
        )
        
        # Load pretrained weights from local file
        if pretrained:
            pretrained_path = "weights/mobilenetv4_conv_large_pretrained.pth"
            if os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded MobileNetV4-Large pretrained weights from {pretrained_path}")
            else:
                print(f"⚠ Warning: Pretrained weights not found at {pretrained_path}")
        
        # Get feature dimension
        self.backbone.eval()
        with torch.no_grad():
            feat_dim = self.backbone(torch.randn(2, 3, 224, 224)).shape[1]
        self.backbone.train()
        
        print(f"MobileNetV4-Large feature dim: {feat_dim}")
        
        self.fc = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return compute_rotation_matrix_from_ortho6d(x)

def mobilenetv4_large_pretrained(pretrained=True, num_classes=6):
    return MobileNetV4LargePretrained(pretrained=pretrained, num_classes=num_classes)
