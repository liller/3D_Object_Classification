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


# 构造整体pointnet网络
class PointNet(nn.Module):
    def __init__(self, classes=40):
        super().__init__()
        self.transform = Transform()
        # 构造全连接层
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    #交叉熵损失
    criterion = torch.nn.NLLLoss()
    #batchsize为outut输出的第一维
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    #如果output在gpu上，需要将以上两个矩阵也移动到gpu上
    if outputs.is_cuda:
        id3x3=id3x3.cuda(gpu)
        id64x64=id64x64.cuda(gpu)
    #这个差异矩阵可以用来惩罚点云旋转矩阵和单位矩阵之间的差异
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    #PointNet的损失函数，由两部分组成。第一部分是NLLLoss，它衡量了输出和标签之间的负对数似然。第二部分是正则化项，它是输入点云旋转和缩放不变性的约束 ？？？
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)


def train(model, train_loader, val_loader=None, epochs=250, save=True):
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    for epoch in range(epochs):
        # 表示模型处于训练模式
        pointnet.train()
        total_loss = total_correct = total_samples = 0.0
        print(f"Epoch {epoch + 1}/{epochs}: ")
        # i为batch的编号，data是当前batch的数据，包含data和label，0表示从编号0开始枚举
        for i, data in enumerate(tqdm(train_loader, 0)):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            # 清楚优化器的梯度值,每个batch都要清楚，避免grad累加
            optimizer.zero_grad()
            # 模型predict输出
            # inputs：bs, channel, length, output:logsoftmax(output), m3x3, m64x64
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()

            # backpropagation
            # 计算当前batch损失值
            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            # 它会根据当前的 loss 计算出所有参数的梯度，并将它们存储在对应的 .grad 属性中。这样在调用 optimizer.step() 时，优化器就会使用这些梯度来更新模型的参数
            loss.backward()
            optimizer.step()

            # print statistics
            # 计算损失和准确率
            total_loss += loss.item()
            total_samples += labels.size(0)
            total_acc = total_correct / total_samples

        train_avg_loss = total_loss / len(train_loader)
        train_avg_acc = total_correct / total_samples
        # 在控制台上显示损失和准确率
        print(f"Epoch {epoch + 1}/{epochs}: Loss: {train_avg_loss:.5f}, Acc: {train_avg_acc:.5f}")
        train_loss.append(train_avg_loss)
        train_acc.append(train_avg_acc)

        # 模型设置为评估模式，
        pointnet.eval()
        total_loss = total_correct = total_samples = 0.0

        # validation
        if val_loader:
            # 用来禁用PyTorch的梯度计算。在验证集或测试集上进行前向推理时，不需要计算梯度（不会反向传播），提高内存效率。
            with torch.no_grad():
                # 分batch循环读取
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)
                    total_correct += (predicted == labels).sum().item()
                    loss = pointnetloss(outputs, labels, m3x3, m64x64)

                    total_loss += loss.item()
                    total_samples += labels.size(0)
                    total_acc = total_correct / total_samples

                val_avg_loss = total_loss / len(val_loader)
                val_avg_acc = total_correct / total_samples
                print(f"Val-Loss: {val_avg_loss:.5f}, Val-Acc: {val_avg_acc:.5f}")
                val_loss.append(val_avg_loss)
                val_acc.append(val_avg_acc)

        if save:
            torch.save(pointnet.state_dict(), "save_" + str(epoch) + ".pth")

    return train_loss, train_acc, val_loss, val_acc
