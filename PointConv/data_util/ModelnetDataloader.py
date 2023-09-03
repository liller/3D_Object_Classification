import numpy as np
import math
import random
import os
import sys
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


import argparse
import datetime
import logging
import importlib
import shutil
import warnings
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):

    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


# 主要负责加载数据，将数据按照[data,label]形式存入dictionary，并存入到本地，方便读取，唯一的预处理是FPS（如果选择了 uniform sampling）。
# 预处理完数据后，每次load的时候需要进行normalize
class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = 1024
        self.process_data = process_data
        self.uniform = False
        self.use_normals = False
        self.num_category = 40

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            # 输出40个类别的名称
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        # rstrip函数移除每行末尾的换行符，
        self.cat = [line.rstrip() for line in open(self.catfile)]
        # zip函数将类别名称与其对应的数字编号打包成元组，再使用dict函数将其转化为字典
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        print(f"self.class:\n {self.classes}\n")

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        # 这行代码是用于检查 split 参数是否为 train 或者 test
        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # 将每个文件的路径存入list中
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        # 如果采用了uniform sampling方法，则保存的文件名中包含"fps"，否则不包含???
        # "modelnet40_train_1024pts.dat"或"modelnet40_train_1024pts_fps.dat"。
        if self.uniform:
            self.save_path = os.path.join(root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        # 是否将数据保存到本地，默认为False
        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    # 加载点云数据，文件格式为以逗号分隔的纯文本格式。np.loadtxt() 函数返回一个 numpy 数组，
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        # 调用FPS抽样
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                #                 print(f"list_of_points{self.list_of_points}")
                #                 print(f"list_of_labels{self.list_of_labels}")
                with open(self.save_path, 'wb') as f:
                    # 将通过FPS或未经过FPS抽样后的数据保存到本地，后续可以直接从本地读取
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    # 获取单个数据样本
    def _get_item(self, index):
        if self.process_data:
            # 本地存储的经过FPS抽样后的数据，获取对应index的xyz和label
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            # 没有没有预处理（FPS抽样）的数据，从本地读取时需要再经过FPS处理
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            # 从本地去加载对应index的数据
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        # 标准化
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            # 只保存前三维坐标信息，将其他例如法向量丢弃，本处为法向量
            point_set = point_set[:, 0:3]

        #         print(f"point_set{point_set}")
        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)
