from models.resnet import resnet18, resnet34, resnet50
from models.mobilenetv2 import mobilenet_v2
from models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from models.mobilenetv4 import mobilenetv4_small, mobilenetv4_medium, mobilenetv4_large
from models.mobilenetv4_pretrained import mobilenetv4_small_pretrained
from models.mobilenetv4_medium_pretrained import mobilenetv4_medium_pretrained
from models.mobilenetv4_large_pretrained import mobilenetv4_large_pretrained
from models.mobilenetv4_hybrid_medium_pretrained import mobilenetv4_hybrid_medium_pretrained
from models.scrfd import SCRFD

__all__ = ["get_model", "SCRFD"]

def get_model(arch, num_classes=6, pretrained=False):
    if arch == "resnet18":
        return resnet18(pretrained=pretrained, num_classes=num_classes)
    elif arch == "resnet34":
        return resnet34(pretrained=pretrained, num_classes=num_classes)
    elif arch == "resnet50":
        return resnet50(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv2":
        return mobilenet_v2(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv3_small":
        return mobilenet_v3_small(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv3_large":
        return mobilenet_v3_large(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv4_small":
        return mobilenetv4_small(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv4_medium":
        return mobilenetv4_medium(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv4_large":
        return mobilenetv4_large(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv4_small_pretrained":
        return mobilenetv4_small_pretrained(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv4_medium_pretrained":
        return mobilenetv4_medium_pretrained(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv4_large_pretrained":
        return mobilenetv4_large_pretrained(pretrained=pretrained, num_classes=num_classes)
    elif arch == "mobilenetv4_hybrid_medium_pretrained":
        return mobilenetv4_hybrid_medium_pretrained(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
