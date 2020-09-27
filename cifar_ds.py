#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 15:43
# @Author  : yulin
# @E-mail  : 844202100@qq.com 
# @School  : bupt
# @File    : cifar_ds.py

from torch.utils.data import Dataset
import numpy as np
import pickle
import cv2
import torchvision.transforms as transforms


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def circle(r):
    D = int(2 * r) + 1
    mask = np.zeros(shape=(D, D, 3), dtype=np.int)
    for i in range(D):
        d = np.sqrt(r * r - (i - r) * (i - r))
        left = int(np.floor(r - d))
        right = int(r + d)
        mask[i, left + 1:right + 1, :] = 1
    return mask


class MyDataset(Dataset):
    def __init__(self, img_path):
        self.mask = circle(15.5)
        self.img_data = unpickle(img_path)[b'data']

    def __len__(self):
        return self.img_data.shape[0]

    def __getitem__(self, idx):
        img = self.img_data[idx].reshape(3, 32, 32)
        img = np.transpose(img, (1, 2, 0))
        img = img * self.mask
        img = img.astype("uint8")
        h, w = img.shape[:2]
        center = (h / 2, w / 2)
        D = np.random.randint(-180, 180 + 1)
        M = cv2.getRotationMatrix2D(center, D, 1.0)
        img = cv2.warpAffine(img, M, (h, w))
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = trans(img)

        return img, float(D / 180)
