# 3D_Object_Classification



## Description
Point cloud data contains rich geometric information and is widely applied in various fields. However, due to its irregular and unordered nature, the analysis of point cloud data presents certain challenges. In this paper, I mainly explore and compare deep learning techniques of point-based methods to achieve point cloud classification. I verify the performance of PointNet, PointNet++, PointConv, and PointMLP models on the ModelNet40 and ScanObjectNN datasets, and enhance the robustness and generalization ability of the models by additionally introducing relevant normal vectors and data augmentation techniques. Experimental results show that the aforementioned four models all demonstrate good classification performance, among which PointMLP achieves state-of-the-art results in multiple tasks, confirm- ing the importance of network depth in point cloud classification. However, it is worth noting that the imbalance of the dataset has a certain impact on the classification results, providing a further direction for the next stage of my research.



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
conda activate cassava_leaf_disease

```

### 2. Prepare Dataset
The dataset could be downloaded from kaggle or run the commands.
```
kaggle competitions download -c cassava-leaf-disease-classification
```
After downloading the datset, place the "train_images" folder in the "Dataset" directory mentioned above。

### 3. Run code

If all the dependencies required for the current project are already installed and placed the dataset in the specific location, you can run main.py 
```
python main.py
```
It could be better to save more time by training the model on GPU, you could specify a target GPU on which to execute the program with the following command
```
CUDA_VISIBLE_DEVICES=" " python main.py 
```
Or if you like using the Notebook to run the code, you could also do the following command before import external library.
```
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ' '
```

The program will read the images from the Datasets directory and automatically create a new folder to store the test set, and then start pre-processing, model building, model training, prediction and evaluation process. 

**Note**: when you copy the datasets to the Datasets directory, you only need to copy the "train_images" folder. The program will automatically divide the test set from the above datasets and create a new directory to store the test data. The ratio of training set, validation set and test set is 8:1:1. 

Due to the monthly limitation of Git LFS uploading large files, it is not possible to upload the trained model files to github, so it will take some time to train all the model. 


## Model Performance & Hyperparamater

### Base Model 
<div align=center>
<img src="https://github.com/liller/AMLSII__22-23_SN21047963/blob/master/Model_architecture/Base_model_3Block.jpg" width="814" height="300">
</div>

| Model Description | Input Size | Optimiser |  LR  |  Class_weight |  Data_Aug | Batch Size  | Epochs | Marco weighted F1-Score | Marco avg F1-Score |
| :----:| :----:| :----: |:----: |:----: | :----: | :----: | :----: | :----: | :----: |
| Base model | (800,600,3) | RMSprop | 0.001 | / | / | 64 |  30 | 65% | 34% |
| Add more layers and droupout | (800,600,3) | RMSprop | 0.001 with ReduceLROnPlateau | / | / | 64 |  30 |  66% | 45% |


### VGG Model 
<div align=center>
<img src="https://github.com/liller/AMLSII__22-23_SN21047963/blob/master/Model_architecture/VGG.jpg">
</div>

| Model Description | Input Size | Optimiser |  LR  |  Class_weight |  Data_Aug | Batch Size  | Epochs | Marco weighted F1-Score | Marco avg F1-Score |
| :----:| :----:| :----: |:----: |:----: | :----: | :----: | :----: | :----: | :----: |
| Feature extractor | (224,224,3) | / | / | / | / | / |  / | 71% | 19% |
| Training from scratch | (224,224,3) | Adam | 2e-5 with ReduceLROnPlateau | / | / | 256 |  30 |  74% | 56% |
| Fine-tune by freezing 2 block | (224,224,3) | Adam | 2e-5 with ReduceLROnPlateau | / | / | 256 |  30 |  79% | 64% |
| Fine-tune by freezing 2 block | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | True | / | 256 |  30 |  67% | 55% |


### ResNet Model 


| Model Description | Input Size | Optimiser |  LR  |  Class_weight |  Data_Aug | Batch Size  | Epochs | Marco weighted F1-Score | Marco avg F1-Score |
| :----:| :----:| :----: |:----: |:----: | :----: | :----: | :----: | :----: | :----: |
| ResNet50 | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | / | 256 |  30 |  75% | 58% |
| ResNet50 | (224,224,3) | Adam | [1.15e-5, 0.001] with lr schedule | / | / | 256 |  30 |  83% | 69% |
| ResNet50 | (224,224,3) | Adam | [1.15e-5, 0.001] with lr schedule | / | True | 256 |  30 |  84% | 70% |
| ResNet50 (Fine-tune) | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | / | 256 |  30 |  74% | 56% |



### Vision Transformer Model 

| Model Description | Input Size | Optimiser |  LR  |  Class_weight |  Data_Aug | Batch Size  | Epochs | Marco weighted F1-Score | Marco avg F1-Score |
| :----:| :----:| :----: |:----: |:----: | :----: | :----: | :----: | :----: | :----: |
| Vit_b32 | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | / | 256 |  30 |  77% | 62% |
| Vit_b32 | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | True | 256 |  50 |  79% | 65% |
| Vit_b32 | (224,224,3) | Adam | [3.03e-7, 1e-5] with lr schedule | True | / | 256 |  50 |  66% | 56% |
| Vit_b16 | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | True | 256 |  30 |  81% | 69% |



### Ensemble Model 

| Model Description | Input Size | Optimiser |  LR  |  Class_weight |  Data_Aug | Batch Size  | Epochs | Marco weighted F1-Score | Marco avg F1-Score |
| :----:| :----:| :----: |:----: |:----: | :----: | :----: | :----: | :----: | :----: |
| Reset50 & ViT_b16 | (224,224,3) | Adam | 3e-6 with ReduceLROnPlateau | / | True | 256 |  50 |  84% | 73% |




