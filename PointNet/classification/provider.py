import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import plotly.graph_objects as go
import plotly.express as px
from path import Path


import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt


# 构造T-net模块，返回一个基于输入batch的点云得到的一个bs个k x k的matrix
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        # 输入通道数为k，输出通道为64，filter大小是1
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        # input.shape == (bs,n,3)
        # (batch_size, channel_size, sequence_length)，bs为batchsize
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        # input为（bs,1024,n）,n为输入序列的长度（点的个数），output为（bs,1024,1）
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        # initialize as identity
        # torch.eye返回一个nxn的单位矩阵
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if xb.is_cuda:
            init = init.cuda(gpu)
        # reshape成（bs,k,k）的tensor
        matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        return matrix


##构造Transform与MLP模块
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        # 分别new出input transform, feature transform
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        # 对应第一个MLP网络，每个点云输入为3，输出为64，filter为1，后面跟一个bn
        self.conv1 = nn.Conv1d(3, 64, 1)

        # 对应第二个MLP网络，每层后面都跟一个bn
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication 将input与matrix相乘，
        # 需要先将input转置，将channel_size, sequence_length维度交换，变为（batch_size，sequence_length，channel_size ）（bs,1024,3）
        # 最后相乘后再将结果转置，保持与input一致
        xb = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        # Flatten
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

