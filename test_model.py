import torch
from models import get_model

print('Testing mobilenetv4_small...')
model = get_model('mobilenetv4_small', pretrained=False, num_classes=6)
x = torch.randn(2, 3, 224, 224)
y = model(x)
print(f'  Output: {y.shape}')
print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
print('OK!')
