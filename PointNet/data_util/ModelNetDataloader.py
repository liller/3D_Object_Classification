from PointNet.data_util.utils import *

path = Path('/scratch/zczqlzh/Dataset/ModelNet/ModelNet40')


folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)};


class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
#         print(folders)
        self.classes = {folder: i for i, folder in enumerate(folders)}
#         print(self.classes)
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
#         print(self.valid)
        self.files = []

        #按照label类别去读取所有点云
        for category in self.classes.keys():
#             print(category)
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    #sample中包含每个点云的存储路径已经对应的label
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)
            print(new_dir)
        print(len(self.files))
#         print(self.files)

    def __len__(self):
        return len(self.files)

    #读取每个点云中的点与面，进行一系列变换，返回sample后的所有点（xyz）
    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            #transforms中PointSampler可以直接调用call函数，需要参数mesh （verts, faces）
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
#         print(pcd_path)
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        #返回一个字典，包含数据（xyz）以及label（category）
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}

train_transforms = transforms.Compose([
                    PointSampler(2048),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])