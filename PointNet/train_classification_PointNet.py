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
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from PointNet.data_util.ModelNetDataloader import *
from PointNet.classification.model.pointnet import *


device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)
inv_classes = {i: cat for cat, i in train_ds.classes.items()};

train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, num_workers=20)
valid_loader = DataLoader(dataset=valid_ds, batch_size=64, num_workers=20)


pointnet = PointNet()
pointnet.to(device)

optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)
train_loss, train_acc, val_loss, val_acc = train(pointnet, train_loader, valid_loader,  save=False)

all_preds = []
all_labels = []
with torch.no_grad():
    for i, data in enumerate(valid_loader):
        print('Batch [%4d / %4d]' % (i + 1, len(valid_loader)))
        inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)

        #         inputs, labels = data['pointcloud'].float(), data['category']
        outputs, __, __ = pointnet(inputs.transpose(1, 2))
        _, preds = torch.max(outputs.data, 1)
        # numpy和cuda不兼容
        all_preds += list(preds.cpu().numpy())
        all_labels += list(labels.cpu().numpy())
    print(inputs)
    print(outputs)


target_names = list(classes.keys())
print(classification_report(all_labels, all_preds, target_names=target_names))

