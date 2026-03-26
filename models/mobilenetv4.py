import torch
import torch.nn as nn
import timm
from utils.general import compute_rotation_matrix_from_ortho6d

class MobileNetV4HeadPose(nn.Module):
    def __init__(self, model_name, pretrained=False, num_classes=6):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.backbone.eval()
        with torch.no_grad():
            feat_dim = self.backbone(torch.randn(2, 3, 224, 224)).shape[1]
        self.backbone.train()
        self.fc = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return compute_rotation_matrix_from_ortho6d(x)

def mobilenetv4_small(pretrained=False, num_classes=6):
    return MobileNetV4HeadPose("mobilenetv4_conv_small", pretrained, num_classes)

def mobilenetv4_medium(pretrained=False, num_classes=6):
    return MobileNetV4HeadPose("mobilenetv4_conv_medium", pretrained, num_classes)

def mobilenetv4_large(pretrained=False, num_classes=6):
    return MobileNetV4HeadPose("mobilenetv4_conv_large", pretrained, num_classes)
