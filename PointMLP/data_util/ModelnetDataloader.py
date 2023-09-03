import numpy as np
import os
import glob
import h5py
from torch.utils.data import Dataset, DataLoader



def random_point_dropout(pc, max_dropout_ratio=0.875):

    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


# 随机平移
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


# 给每个点加上噪音
def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def load_data(partition):
    #     download()
    BASE_DIR = os.path.dirname('/scratch/zczqlzh/Dataset/ModelNet/')
    #     DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    # 按照文件依次读取，该数据集中每个经过预处理的包含2048个点，只有xyz特征
    for h5_name in glob.glob(os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        # f为一个点云文件，包含2048个点云，（ply_data_train4只包含1648个点云）
        f = h5py.File(h5_name, 'r')
        # 读取的文件中的点云数据，2048个点云，每个点云2048个点，每个点包含xyz特征 data.shape = (2048, 2048, 3)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()

        # all_data是一个list，循环将多个文件的点云存入一个list，长度为9840
        all_data.append(data)
        all_label.append(label)
    # 相当于将最外层的维度去掉，(9840, 2048, 3)
    all_data = np.concatenate(all_data, axis=0)
    # (9840, 1)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]