"""
MIT License

Copyright (c) 2019 Boshen Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
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


class DepthRegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, feature_size=256):
        super(DepthRegressionModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)

        # out is B x C x W x H, with C = 3*num_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes)


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, feature_size=256):
        super(RegressionModel, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes * 2, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)

        # out is B x C x W x H, with C = 3*num_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes, 2)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes, 2)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNetBackBone(nn.Module):
    def __init__(self, light):
        super(ResNetBackBone, self).__init__()
        if light:
            self.model = resnet.resnet18(pretrained=True)
        else:
            self.model = resnet.resnet50(pretrained=True)

    def forward(self, x):
        n, c, h, w = x.size()  # x: [B, 1, H ,W]

        x = x[:, 0:1, :, :]  # depth
        x = x.expand(n, 3, h, w)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)

        return x3, x4


def generate_anchors(P_h=None, P_w=None):
    if P_h is None:
        P_h = np.array([2,6,10,14])

    if P_w is None:
        P_w = np.array([2,6,10,14])

    num_anchors = len(P_h) * len(P_h)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k,1] = P_w[j]
            anchors[k,0] = P_h[i]
            k += 1
    return anchors


def shift(shape, stride, anchors):
    shift_h = np.arange(0, shape[0]) * stride
    shift_w = np.arange(0, shape[1]) * stride

    shift_h, shift_w = np.meshgrid(shift_h, shift_w)
    shifts = np.vstack((shift_h.ravel(), shift_w.ravel())).transpose()

    # add A anchors (1, A, 2) to
    # cell K shifts (K, 1, 2) to get
    # shift anchors (K, A, 2)
    # reshape to (K*A, 2) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 2))
    # print(all_anchors.shape)
    # print(all_anchors[:32])

    return all_anchors


class A2J_model(nn.Module):
    def __init__(self, num_classes, P_h=None, P_w=None, shape=[11, 11], stride=16, dropout_rate=0., is_3D=True,
                 light=False):
        super(A2J_model, self).__init__()
        self.dropout_rate = dropout_rate
        self.is_3D = is_3D
        self.light = light
        anchors = generate_anchors(P_h=P_h, P_w=P_w)
        self.all_anchors = torch.from_numpy(shift(shape, stride, anchors)).float() #(w*h*A)*2
        self.Backbone = ResNetBackBone(light)  # 1 channel depth only
        if light:
            self.regressionModel = RegressionModel(512, num_classes=num_classes)
            self.classificationModel = ClassificationModel(256, num_classes=num_classes)
            self.dropout = nn.Dropout(dropout_rate)
            if is_3D:
                self.DepthRegressionModel = DepthRegressionModel(512, num_classes=num_classes)
        else:
            self.regressionModel = RegressionModel(2048, num_classes=num_classes)
            self.classificationModel = ClassificationModel(1024, num_classes=num_classes)
            self.dropout = nn.Dropout(dropout_rate)
            if is_3D:
                self.DepthRegressionModel = DepthRegressionModel(2048, num_classes=num_classes)

    def forward(self, x):
        anchor = self.all_anchors.to(x.device)
        x3, x4 = self.Backbone(x)
        x3 = self.dropout(x3)
        x4 = self.dropout(x4)
        classification = self.classificationModel(x3) # N*(w/16*h/16*A)*P
        regression = self.regressionModel(x4) # N*(w/16*h/16*A)*P*2
        reg_weight = F.softmax(classification, dim=1)  # N*(w/16*h/16*A)*P
        reg_weight_xy = torch.unsqueeze(reg_weight, 3).expand(
            reg_weight.shape[0], reg_weight.shape[1], reg_weight.shape[2], 2)  # N*(w/16*h/16*A)*P*2
        anchor_joints_2d = (reg_weight_xy * torch.unsqueeze(anchor, 1)).sum(1) # N*P*2
        # anchor_joints_2d[..., 0], anchor_joints_2d[..., 1] = anchor_joints_2d[..., 1], anchor_joints_2d[..., 0]

        reg = torch.unsqueeze(anchor, 1) + regression  # N*(w/16*h/16*A)*P*2
        regression_joints_2d = (reg_weight_xy*reg).sum(1) # N*P*2
        # regression_joints_2d[..., 0], regression_joints_2d[..., 1] = \
        #     regression_joints_2d[..., 1], regression_joints_2d[..., 0]

        if self.is_3D:
            depthregression = self.DepthRegressionModel(x4) # N*(w/16*h/16*A)*P
            depth_value = (reg_weight * depthregression).sum(1)
            return classification, regression, depthregression, anchor_joints_2d, regression_joints_2d, depth_value
        return classification, regression, anchor_joints_2d, regression_joints_2d


if __name__ == "__main__":
    num_classes = 14
    w, h = 176, 176
    B = 10
    depth = torch.rand([B, 1, h, w], dtype=torch.float32).cuda()
    model = A2J_model(num_classes).cuda()
    anchor_joints_2d, regression_joints_2d, depth_value = model(depth)
    print(anchor_joints_2d.shape)
    print(regression_joints_2d.shape)
    print(depth_value.shape)
