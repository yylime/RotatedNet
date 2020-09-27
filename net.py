#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 17:27
# @Author  : yulin
# @E-mail  : 844202100@qq.com 
# @School  : bupt
# @File    : net.py
from torchvision import models
import torch.nn as nn


def cnn():
    cnn = models.resnet18(pretrained=True)
    cnn.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1))
    return cnn
