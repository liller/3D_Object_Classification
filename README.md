# 3D_Object_Classification



## Description
Point cloud data contains rich geometric information and is widely applied in various fields. However, due to its irregular and unordered nature, the analysis of point cloud data presents certain challenges. In this paper, I mainly explore and compare deep learning techniques of point-based methods to achieve point cloud classification. I verify the performance of PointNet, PointNet++, PointConv, and PointMLP models on the ModelNet40 and ScanObjectNN datasets, and enhance the robustness and generalization ability of the models by additionally introducing relevant normal vectors and data augmentation techniques. Experimental results show that the aforementioned four models all demonstrate good classification performance, among which PointMLP achieves state-of-the-art results in multiple tasks, confirming the importance of network depth in point cloud classification. However, it is worth noting that the imbalance of the dataset has a certain impact on the classification results, providing a further direction for the next stage of my research.


## Results
The Shape Classification on ModelNet40 results are shown in the following table:

| Method | Inputs | OA(%) | mAcc(%) | F1-Score(%) | Precision(%) | Recall(%) |
| :----:| :----:| :----: |:----: |:----: |:----: |:----: |
| PointNet | 2048 P | 89.0 | 86.1 | - | - | - |
| PointNet++ | 1024 P | 91.7 | 89.0 | 91.8 | 92.1 | 91.7 |
|  | 1024 P+N | 92.4 | 89.9 | 92.5 | 92.6 | 92.4 |
| PointConv | 1024 P | 91.7 | 88.5 | 91.8 | 92.0 | 91.7 |
|  | 1024 P+N | 92.1 | 89.9 | 92.2 | 92.4 | 92.1 |
| PointMLP | 1024 P | 93.4 | 91.3 | 93.4 | 93.5 | 93.4 |
|  | 1024 P+N | 94.1 | 91.4 | 94.2 | 94.4 | 94.1 |

The Shape Classification on ScanObjectNN results are shown in the following table:

| Method | Inputs | OA(%) | mAcc(%) | F1-Score(%) | Precision(%) | Recall(%) |
| :----:| :----:| :----: |:----: |:----: |:----: |:----: |
| PointNet++ | 1024 P | 79.5 | 77.9 | 79.1 | 79.1 | 79.5 |
| PointConv | 1024 P | 77.9 | 75.1 | 77.9 | 78.5 | 77.9 |
| PointMLP | 1024 P | 86.6 | 84.5 | 86.5 | 86.7 | 86.6 |

Specific hyperparameters and techniques can be found at the end of the instruction. 


## Role of each file
The current project structure is shown below
```
.
├── PointConv
│   ├── classification
│   │   ├── model
│   │   │   └── PointConv.py
│   │   └── provider.py
│   ├── data_util
│   │   ├── ModelnetDataloader.py
│   │   └── ScanObjectDataloader.py
│   └── train_classification_PointConv.py
├── PointMLP
│   ├── classification
│   │   ├── models
│   │   │   └── PointMLP.py
│   │   └── provider.py
│   ├── data_util
│   │   ├── ModelnetDataloader.py
│   │   └── ScanObjectDataloader.py
│   ├── train_classification_PointMLP.py
│   └── utils.py
├── PointNet
│   ├── classification
│   │   ├── model
│   │   │   └── pointnet.py
│   │   └── provider.py
│   ├── data_util
│   │   ├── ModelNetDataloader.py
│   │   └── utils.py
│   └── train_classification_PointNet.py
└── PointNet2
    ├── classification
    │   ├── models
    │   │   └── Pointnet2_ssg.py
    │   └── provider.py
    ├── data_util
    │   ├── ModelnetDataloader.py
    │   └── ScanObjectDataloader.py
    └── train_classification_PointNet++.py

```




## Getting started
### 1. Setup
1. Create a new virtual conda environment based on the provided environment.yml file and execute the following statement in the project path. 

```
conda env create -f environment.yml
```
2. Activate this conda virtual environment. 
```
conda activate point-cloud

```


### 2. Run code

If all the dependencies required for the current project are already installed and downloaded the dataset in the specific location, you can run train_classification_Pointxxx.py 
```
python train_classification_Pointxxx.py 
```
It could be better to save more time by training the model on GPU, you could specify a target GPU on which to execute the program within the code.

Due to the monthly limitation of Git LFS uploading large files, it is not possible to upload the trained model files to github, so it will take some time to train all the model. 



