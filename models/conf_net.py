import torch.nn as nn
from torch.nn import init
import torch
import torch.nn.functional as F
import numpy as np

import os
import sys
dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(dir)
from models import resnet


class ConfNet(nn.Module):
    def __init__(self, num_views, dropout_rate):
        super(ConfNet, self).__init__()
        self.resnet = resnet.resnet18(pretrained=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1000, num_views)

    def forward(self, x):
        """

        :param x: Tensor(B, 1, 176, 176)
        :return:
        """
        n, c, h, w = x.size()  # x: [B, 1, H ,W]

        x = x[:, 0:1, :, :]  # depth
        x = x.expand(n, 3, h, w)
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    confnet = ConfNet(25, 0.5)
    input = torch.randn((4, 1, 176, 176), dtype=torch.float32)
    output = confnet(input)
    print(output.shape)