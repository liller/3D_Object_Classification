
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


#输出为【1024，6】
def farthest_point_sample_first(point, npoint):

    #shape = [1000,6]
    N, D = point.shape
    #取前三维，剩下的三维为法向量丢弃，【1000，3】
    xyz = point[:,:3]
    #新建一个array大小为1024，代表要抽样出的1024个中心点
    centroids = np.zeros((npoint,))
    #创建了一个大小为N的numpy数组，每个元素初始化为1e10。这个数组可能用于存储每个点到当前点的最短距离，初始值为一个非常大的值，表示当前点到其他所有点的距离都很远，还没有被更新过。
    #在计算最短距离的过程中，如果发现一个点到当前点的距离更短，就会更新这个数组中对应位置的值。最后，这个数组中存储的就是每个点到当前点的最短距离。
    distance = np.ones((N,)) * 1e10
    #是用于生成从0到N（不包括N）之间的一个随机整数，该数代表的点作为第一个选中的中心点，N为一个点云中所有的点数，当前最大为10000
    farthest = np.random.randint(0, N)
    #最终得到的中心点集即为具有最大空间分离度的npoint个点。
    for i in range(npoint):
        centroids[i] = farthest
        #通过切片，取出当前点的xyz
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
#     print(f"after sampling, the index of all points: {point}")
#     print(f"The shape the index of all points: {point.shape}")

    return point


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def load_scanobjectnn_data(partition):
#     download()
    BASE_DIR = os.path.dirname('/scratch/zczqlzh/Dataset/ScanObjectNN/')
    all_data = []
    all_label = []

    #按照文件依次读取，该数据集中每个经过预处理的包含2048个点，只有xyz特征
    h5_name = BASE_DIR + '/ScanObjectNN/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='training'):
        self.data, self.label = load_scanobjectnn_data(partition)
        print(f"self.class:\n {self.label}\n")

        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = farthest_point_sample_first(self.data[item], self.num_points)
        #         pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        pointcloud[:, 0:3] = pc_normalize(pointcloud[:, 0:3])

        #         if self.partition == 'training':
        #             pointcloud = translate_pointcloud(pointcloud)
        #             np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


# log_string('==> Preparing data..')
train_dataset = ScanObjectNN(partition='training', num_points=args.num_points)
test_dataset = ScanObjectNN(partition='test', num_points=args.num_points)


def read_off(file):
    if 'OFF' != file.readline().strip():
        #         print(file)
        raise ValueError('Not a valid OFF header')

    #         raise('Not a valid OFF header')
    # 读取每个点坐标，共n_verts行，每个点的坐标存储在verts列表中
    # 读取每个面的顶点索引，共n_faces行，每个点的顶点索引存储在faces列
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    # 忽略第一列的顶点数信息
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    # 用于在三角形中采样一个点，终返回一个包含采样点坐标的三元组tuple (f(0), f(1), f(2))
    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        #         print(type((f(0), f(1), f(2))))
        # 返回一个元组，是一个静态列表
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        # 计算所有面的面积
        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
        # 用于从给定序列中进行随机采样，给定的序列是 faces，每个面的面积area作为权重，k 参数指定了采样的数量
        sampled_faces = (random.choices(faces,
                                        weights=areas,
                                        cum_weights=None,
                                        k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


class Normalize(object):
    # 将对象当作函数来使用，即调用该对象时，就会直接执行call函数
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        # 计算出点云中所有点的欧几里得范数,并除以最大的范数
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)


def default_transforms():
    #每个点云sample出1024个点
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
#                                 ToTensor()
                              ])


# data = f['data'][:].astype('float')

path = Path('/scratch/zczqlzh/Dataset/ModelNet/ModelNet40')


class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        self.classes = {'bed': 2, 'chair': 8, 'desk': 12, 'door': 13, 'sink': 29, 'sofa': 30, 'table': 33, 'toilet': 35}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []

        # 按照label类别去读取所有点云
        for category in self.classes.keys():
            new_dir = root_dir / Path(category) / folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    # sample中包含每个点云的存储路径已经对应的label
                    sample['pcd_path'] = new_dir / file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    # 读取每个点云中的点与面，进行一系列变换，返回sample后的所有点（xyz）
    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            # transforms中PointSampler可以直接调用call函数，需要参数mesh （verts, faces）
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        #         print(pcd_path)
        category = self.files[idx]['category']
        #         print(category)
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)

        point_set = pointcloud.astype('float32')
        #         print(point_set)
        #         data = f['data'][:].astype('float')
        #         print(point_set.shape)
        #         label = self.classes[category]
        #         print(f"before{idx}:{label}\n")
        self.classes = {'bed': 1, 'chair': 5, 'desk': 6, 'door': 8, 'sink': 11, 'sofa': 12, 'table': 13, 'toilet': 14}
        label1 = self.classes[category]
        #         print(f"after{idx}:{label1}\n")

        return point_set, label1
#


train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
#                     ToTensor()
                    ])

train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)


from torch.utils.data import ConcatDataset, DataLoader, Dataset

# 假设 dataset1 和 dataset2 是两个 Dataset 对象
train_dataset1 = ConcatDataset([train_dataset, train_ds])
test_dataset1 = ConcatDataset([test_dataset, valid_ds])

train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=16, shuffle=True, num_workers=10, drop_last=True)
valid_loader1 = torch.utils.data.DataLoader(dataset=test_dataset1, batch_size=16, shuffle=False, num_workers=10)