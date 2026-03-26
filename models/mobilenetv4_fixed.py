import torch
import torch.nn as nn
import timm

class MobileNetV4HeadPose(nn.Module):
    """MobileNetV4 backbone for 6D rotation representation head pose estimation"""
    
    def __init__(self, model_name, pretrained=False, num_classes=6):
        super().__init__()
        # Load backbone without classifier
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        # Get feature dimension in eval mode
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            feat = self.backbone(dummy)
            feat_dim = feat.shape[1]
        self.backbone.train()
        
        # Head for 6D rotation representation
        self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


def mobilenetv4_small(pretrained=False, num_classes=6):
    return MobileNetV4HeadPose('mobilenetv4_conv_small', pretrained=pretrained, num_classes=num_classes)

def mobilenetv4_medium(pretrained=False, num_classes=6):
    return MobileNetV4HeadPose('mobilenetv4_conv_medium', pretrained=pretrained, num_classes=num_classes)

def mobilenetv4_large(pretrained=False, num_classes=6):
    return MobileNetV4HeadPose('mobilenetv4_conv_large', pretrained=pretrained, num_classes=num_classes)

def mobilenetv4_hybrid_medium(pretrained=False, num_classes=6):
    return MobileNetV4HeadPose('mobilenetv4_hybrid_medium', pretrained=pretrained, num_classes=num_classes)

def mobilenetv4_hybrid_large(pretrained=False, num_classes=6):
    return MobileNetV4HeadPose('mobilenetv4_hybrid_large', pretrained=pretrained, num_classes=num_classes)
