#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/9/16
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import Flatten

from backbone import resnet18


class YOLO(nn.Module):
    def __init__(self, num_bboxes=2, num_classes=20):
        super(YOLO, self).__init__()
        self.feature_size = 7
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.backbone = resnet18(pretrained=True)

        self.fc_layers = self._make_fc_layers()

    def forward(self, x, target=None):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        x = self.backbone(x)[-1]
        x = self.fc_layers(x)

        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        x = x.view(x.size(0), 5 * B + self.num_classes, -1).permute(0, 2, 1)
        return x

    def _make_fc_layers(self):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        net = nn.Sequential(
            Flatten(),

            nn.Linear(14 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid()
        )

        return net
