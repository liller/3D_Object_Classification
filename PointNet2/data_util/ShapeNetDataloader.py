import numpy as np
import math
import random
import os
import sys
import glob
import torch
import json
import h5py
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

from torch.autograd import Variable
import plotly.graph_objects as go
import plotly.express as pxz
from path import Path
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import torch.backends.cudnn as cudnn
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


'''DATA LOADING'''
# log_string('Load dataset ...')
data_path = ''

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


# 该数据集有16个大类，每个大类中可以分割成不同的小类，总共50个小类别，每个数据存放文件夹有7列，xyz+ surface normals + 分割的小的类别label（50）
class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500,
                 split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        # 该文件记录了不同大类对应的文件夹（16类）
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        # 这个字典的键是分类的名称，值是对应的索引。比如如果有3个类别，那么就会形成 {类别1:0, 类别2:1, 类别3:2} 这样的一个字典
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            # 原始的数据是划分成3部分的，将其original的train和val合并成一个文件夹
            # fn[0:-4] 是去除文件名末尾的四个字符，常见的情况是为了去除文件的扩展名 .txt
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                # fns 中属于训练集的文件名筛选出来
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                # 按照大类存放对应的路径
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        # self.cat.keys()大类的label
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, cls, seg = self.cache[index]
        else:
            # 文件名fn和分类cat，cls是类别标签
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            # data为一个点云文件，二维数组
            data = np.loadtxt(fn[1]).astype(np.float32)

            point_set = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)

            #             if not self.normal_channel:
            #                 point_set = data[:, 0:3]
            # #                 print(point_set.shape)
            #             else:
            #                 point_set = data[:, 0:6]
            #             #取出data的最后一列，并将其转换为32位整数类型。
            #             seg = data[:, -1].astype(np.int32)

            if len(self.cache) < self.cache_size:
                # self.cache的字典中，每个index中的内容是一个元组，这个元组包含三个元素：point_set, cls, 和 seg。
                self.cache[index] = (point_set, normal, seg, cls)
        #                 self.cache[index] = (point_set, cls, seg)
        #         point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        point_set = pc_normalize(point_set)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #         print(len(choice))
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]

        return point_set, cls, seg, normal

    def __len__(self):
        return len(self.datapath)


num_classes = 16
num_part = 50