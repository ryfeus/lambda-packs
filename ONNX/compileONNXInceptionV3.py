import torch
from torchvision import models

model = models.inception_v3(pretrained=True)
torch.onnx.export(model, torch.randn(1,3,299, 299), 'inception_v3.onnx')