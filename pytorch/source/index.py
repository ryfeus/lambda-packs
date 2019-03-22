import torchvision.models as models
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch
import torch.nn as nn

def handler(event, context):
    return 0