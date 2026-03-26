import torch
import torch.nn as nn
import timm
from utils.general import compute_rotation_matrix_from_ortho6d

class MobileNetV4HeadPosePretrained(nn.Module):
    """MobileNetV4 with ImageNet pretrained weights"""
    def __init__(self, model_name, pretrained=True, num_classes=6):
        super().__init__()
        # Load pretrained backbone from local file or timm
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="avg")
        
        # Load pretrained weights
        if pretrained:
            try:
                # Try loading from local file first
                state_dict = torch.load("weights/mobilenetv4_conv_small_pretrained.pth", map_location="cpu")
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights from local file")
            except Exception as e:
                print(f"Could not load pretrained weights: {e}")
        
        self.backbone.eval()
        with torch.no_grad():
            feat_dim = self.backbone(torch.randn(2, 3, 224, 224)).shape[1]
        self.backbone.train()
        
        self.fc = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return compute_rotation_matrix_from_ortho6d(x)

def mobilenetv4_small_pretrained(pretrained=True, num_classes=6):
    return MobileNetV4HeadPosePretrained("mobilenetv4_conv_small", pretrained=pretrained, num_classes=num_classes)
