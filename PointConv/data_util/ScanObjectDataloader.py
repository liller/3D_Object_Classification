
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import h5py

#输出为【1024，6】
def farthest_point_sample_first(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D] [10000,6]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D] [1024,6](只是为了筛选符合要求的1024个点，因此没有丢弃法向量)
    """
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
    label = f['label'][:].astype('int64')
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