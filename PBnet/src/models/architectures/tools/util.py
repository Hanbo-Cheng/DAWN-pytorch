from torch import nn

import torch.nn.functional as F
import torch

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d
from src.models.architectures.tools.resnet import resnet34

class MyResNet34(nn.Module):
    def __init__(self,embedding_dim,input_channel = 3):
        super(MyResNet34, self).__init__()
        self.resnet = resnet34(norm_layer = BatchNorm2d,num_classes=embedding_dim,input_channel = input_channel)
    def forward(self, x):
        return self.resnet(x)