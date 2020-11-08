#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 18:33
# @Author  : yulin
# @E-mail  : 844202100@qq.com 
# @School  : bupt
# @File    : main.py
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ds import MyDataset, Scenery
from net import cnn
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--bs", default=32, type=int, help="batch size")
parser.add_argument('-d', "--dataset", default='scenery', type=str, help="type of dataset")
parser.add_argument('-o', "--model_name", default='best', type=str, help="model name")
args = parser.parse_args()
epochs = args.epochs
bs = args.bs
ds = args.dataset
model_name = args.model_name

if __name__ == '__main__':
    # 判断gpu是否可用
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)
    # 构建数据读取
    if ds == 'cifar':
        path = ['imgs/data_batch_1', 'imgs/data_batch_2', 'imgs/data_batch_3', 'imgs/data_batch_4', 'imgs/data_batch_5']
        val_path = 'imgs/test_batch'
        train_dst = MyDataset(path)
        valid_dst = MyDataset(val_path)
    if ds == 'scenery':
        path = 'dataset'
        img_path = [os.path.join(path, i) for i in os.listdir(path)]
        split = len(img_path) // 10
        train_dst = Scenery(img_path[:-split])
        valid_dst = Scenery(img_path[-split:])

    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=bs, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dst, batch_size=bs, shuffle=False, pin_memory=True)

    # 模型
    cnn = cnn()
    cnn.to(device)
    # print(cnn)
    # 配置
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    # optimizer = optim.SGD(cnn.parameters(), lr=cfg.lr, momentum=0.9, nesterov=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)

    # 训练
    score = float('inf')
    for e in range(epochs):
        losses = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = cnn(inputs)
            loss = loss_fn(outputs.flatten().float(), targets.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()

        print("Step:%d train-Loss is %f:" % (e, losses))

        # 测试
        val_loss = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(valid_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = cnn(inputs)
                loss = loss_fn(outputs.flatten().float(), targets.float())
                val_loss += loss
            print("Step:%d valid-Loss is %f:" % (e, val_loss))

        if val_loss < score:
            score = val_loss
            torch.save(cnn, 'outputs/%s.pth' % model_name)
            print("Best step is %d" % (e))
