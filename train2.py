"""

https://www.kaggle.com/code/richolson/isic-2024-imagenet-train-oof-preds-public

"""


# ============================== Import Required Libraries ==============================

import os  
import gc
import cv2
import math
import copy
import time
import random
import glob
from matplotlib import pyplot as plt
import h5py
from PIL import Image
from io import BytesIO

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda import amp
import torchvision
from torcheval.metrics.functional import binary_auroc

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold 
from sklearn.model_selection import GroupKFold

from sklearn.metrics import roc_curve

# For Image Models
import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
# 会导致多个GPU无法并行计算
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.nn.parallel import DataParallel


# ============================== Training Configuration ==============================


CONFIG = {
    "seed": 42,
    "epochs": 20,

    "img_size": 336,
    "model_name": "eva02_small_patch14_336.mim_in22k_ft_in1k",
    # "img_size": 384,
    # "model_name": "vit_base_patch16_clip_384.openai_ft_in12k_in1k",

    # 164:only kaggle_data
    # 96:only github_data
    "train_batch_size": 164, # 96 32

    "valid_batch_size": 164, # 64
    "scheduler": 'CosineAnnealingLR',
    # "checkpoint": '/home/xyli/kaggle/Kaggle_ISIC/vit/AUROC0.5322_Loss0.2527_epoch3.bin',
    "checkpoint": '/home/xyli/kaggle/Kaggle_ISIC/eva/AUROC0.8170_Loss0.7140_epoch12.bin',
    # "checkpoint": None,

    # 手动调节学习率
    "learning_rate": 1e-5, # 1e-5
    "min_lr": 1e-6, # 1e-6
    "T_max": 20,

    "weight_decay": 1e-6,
    "fold" : 0,
    "n_fold": 5,
    "n_accumulate": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])


ROOT_DIR = "/home/xyli/kaggle"
HDF_FILE = f"{ROOT_DIR}/train-image.hdf5"


# ============================== Read the Data ==============================


df = pd.read_csv(f"{ROOT_DIR}/train-metadata.csv")

# print("        df.shape, # of positive cases, # of patients")
# print("original>", df.shape, df.target.sum(), df["patient_id"].unique().shape)

# df_positive = df[df["target"] == 1].reset_index(drop=True) # 取出target=1的所有行
# df_negative = df[df["target"] == 0].reset_index(drop=True) # 取出target=0的所有行

# # 从2个数据集中各自以 positive:negative = 1:20 进行采样，我感觉是确保验证集中正负样本比例为1:20
# df = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*20, :]])  
# print("filtered>", df.shape, df.target.sum(), df["patient_id"].unique().shape)

# df = df.reset_index(drop=True)

# print(df.shape[0], df.target.sum())

# 用于计算一个学习率调整器的一个参数
# 因为之后要合并数据集,算了一下合并后大约是合并前2.4倍,合并前是8k,合并后是20k左右
# CONFIG['T_max'] = 2.4*df.shape[0] * (CONFIG["n_fold"]-1) * CONFIG['epochs'] // CONFIG['train_batch_size'] // CONFIG["n_fold"]
# print(CONFIG['T_max'])


# sgkf = StratifiedGroupKFold(n_splits=CONFIG['n_fold'])
# for fold, ( _, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
#       df.loc[val_ , "kfold"] = int(fold)


gkf = GroupKFold(n_splits=5)
df["kfold"] = -1
for idx, (train_idx, val_idx) in enumerate(gkf.split(df, df["target"], groups=df["patient_id"])):
    df.loc[val_idx, "kfold"] = idx

""" 
统计一下各折的信息 

Fold 0: 206 patients
Fold 1: 209 patients
Fold 2: 208 patients
Fold 3: 209 patients
Fold 4: 210 patients
Total patients: 1042
"""
# Add summary
fold_summary = df.groupby("kfold")["patient_id"].nunique().to_dict()
total_patients = df["patient_id"].nunique()
print(f"Fold Summary (patients per fold):")
for fold, count in fold_summary.items():
    if fold != -1:  # Exclude the initialization value
        print(f"Fold {fold}: {count} patients")
print(f"Total patients: {total_patients}")




""" 统计一下数据集总体信息 """
print("\nOriginal Dataset Summary:")
print(f"Total number of samples: {len(df)}")
print(f"Number of unique patients: {df['patient_id'].nunique()}")
original_positive_cases = df['target'].sum()
original_total_cases = len(df)
original_positive_ratio = original_positive_cases / original_total_cases
print(f"Number of positive cases: {original_positive_cases}")
print(f"Number of negative cases: {original_total_cases - original_positive_cases}")
print(f"Ratio of negative to positive cases: {(original_total_cases - original_positive_cases) / original_positive_cases:.2f}:1")



"""  
Downsample 
Keeping just 1% of negatives!

Balanced Dataset Summary:
Total number of samples: 401059
Number of unique patients: 1042
Number of positive cases: 393
Number of negative cases: 4007
New ratio of negative to positive cases: 10.20:1
"""
df_train = df
#keep all positives
df_target_1 = df_train[df_train['target'] == 1]
#just use 1% of negatives
df_target_0 = df_train[df_train['target'] == 0].sample(frac=0.01, random_state=42)
df_train_balanced = pd.concat([df_target_1, df_target_0]).reset_index(drop=True)
# Print balanced dataset summary
print("")
print("Balanced Dataset Summary:")
print(f"Total number of samples: {len(df_train)}")
print(f"Number of unique patients: {df_train['patient_id'].nunique()}")
positive_cases = df_train_balanced['target'].sum()
total_cases = len(df_train_balanced)
positive_ratio = positive_cases / total_cases
print(f"Number of positive cases: {positive_cases}")
print(f"Number of negative cases: {total_cases - positive_cases}")
print(f"New ratio of negative to positive cases: {(total_cases - positive_cases) / positive_cases:.2f}:1")


# from IPython import embed
# embed()
# ============================== Dataset Class ==============================
class ISICDataset(Dataset):
    def __init__(self, hdf5_file, isic_ids, targets=None, transform=None):
        self.hdf5_file = hdf5_file
        self.isic_ids = isic_ids
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            img_bytes = f[self.isic_ids[idx]][()]
        
        img = Image.open(BytesIO(img_bytes))
        img = np.array(img)  # Convert PIL Image to numpy array
        
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        
        if self.targets is not None:
            target = self.targets[idx]
        else:
            target = torch.tensor(-1)  # Dummy target for test set
            
        return img, target

# ============================== Create Model ==============================

class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(ISICModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        return self.sigmoid(self.model(images))
    


if CONFIG['checkpoint'] is not None:
    model = ISICModel(CONFIG['model_name'], pretrained=False)

    checkpoint = torch.load(CONFIG['checkpoint'])
    print(f"load checkpoint: {CONFIG['checkpoint']}") 
    # 去掉前面多余的'module.'
    new_state_dict = {}
    for k,v in checkpoint.items():
        new_state_dict[k[7:]] = v
    model.load_state_dict( new_state_dict )
else:
    model = ISICModel(CONFIG['model_name'], pretrained=True)

model = model.cuda() 
# model.to(CONFIG['device'])

model = DataParallel(model) 


# ============================== Augmentations ==============================
# Prepare augmentation
aug_transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.5),
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

base_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


#--------------------------------------------测试一下数据增强效果
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def visualize_augmentations_positive(dataset, num_samples=3, num_augmentations=5, figsize=(20, 10)):
    # Find positive samples
    positive_samples = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == 1:  # Assuming 1 is the positive class
            positive_samples.append(i)

        if len(positive_samples) == num_samples:
            break
    
    if len(positive_samples) < num_samples:
        print(f"Warning: Only found {len(positive_samples)} positive samples.")
    
    fig, axes = plt.subplots(num_samples, num_augmentations + 1, figsize=figsize)
    fig.suptitle("Original and Augmented Versions of Positive Samples", fontsize=16)

    for sample_num, sample_idx in enumerate(positive_samples):
        # Get a single sample
        original_image, label = dataset[sample_idx]
        
        # If the image is already a tensor (due to ToTensorV2 in the transform), convert it back to numpy
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.permute(1, 2, 0).numpy()
            
        # Reverse the normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_image = (original_image * std + mean) * 255
        original_image = original_image.astype(np.uint8)

        # Display original image
        axes[sample_num, 0].imshow(original_image)
        axes[sample_num, 0].axis('off')
        axes[sample_num, 0].set_title("Original", fontsize=10)

        # Apply and display augmentations
        for aug_num in range(num_augmentations):
            augmented = dataset.transform(image=original_image)['image']
            # If the result is a tensor, convert it back to numpy
            if isinstance(augmented, torch.Tensor):
                augmented = augmented.permute(1, 2, 0).numpy()
            # Reverse the normalization
            augmented = (augmented * std + mean) * 255
            augmented = augmented.astype(np.uint8)
            
            axes[sample_num, aug_num + 1].imshow(augmented)
            axes[sample_num, aug_num + 1].axis('off')
            axes[sample_num, aug_num + 1].set_title(f"Augmented {aug_num + 1}", fontsize=10)

    plt.tight_layout()
    plt.show()
    
augtest_dataset = ISICDataset(
    hdf5_file=HDF_FILE,
    isic_ids=df_train['isic_id'].values,
    targets=df_train['target'].values,
    transform=aug_transform,
)

visualize_augmentations_positive(augtest_dataset)

from IPython import embed
embed()
# ============================== cutmix+mixup ==============================

def rand_bbox(size, lam):
    """
    返回原图1-lam倍的区域
    args:
        size: 是图片的shape，即（b,c,w,h）
        lam: 调整融合区域的大小，lam 越接近 1，融合框越小，融合程度越低
    return:
        bbx1, bby1, bbx2, bby2: 返回一个随机生成的矩形框，用于确定两张图像的融合区域
    """
    W = size[2]
    H = size[3]

    # sqrt是要保证 cut_w*cut_h = (1-lam)*W*H
    cut_rat = np.sqrt(1. - lam)
    
    cut_w = int(W * cut_rat) # 裁剪总w长度
    cut_h = int(H * cut_rat) # 裁剪总h长度

    # 随便选裁剪的一个中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # 因为限定0~w内，所以会导致误差，使得cut_w*cut_h < (1-lam)*W*H
    bbx1 = np.clip(cx - cut_w // 2, 0, W) # bbx1限定在0~W内
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets1, alpha):
    """
    args:
        data: 训练数据，data.shape=(b,c,w,h)  
        targets1: 训练数据标签，targets1.shape=(b)
        alpha: 生成一个什么样的数据分布，lam就是从这个分布中随机取值
    return:
        data: 经过cutmix改变后的训练数据
        targets：[targets1, shuffled_targets1, lam]
            targets1: 被cut图像的标签
            shuffled_targets1: cut区域外来图的标签
            lam: 由于np.clip导致误差，这个是消除误差调整后的lam           
    """
    
    # 对b这个维度进行随机打乱,产生随机序列indices
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices] # 这是打乱b后的数据,shape=(b,c,w,h)
    shuffled_targets1 = targets1[indices] # 打乱后的标签，shape=(b,)
    
    # 在alpha生成的分布中随机抽1个值lam，它控制了两个图像的融合区域大小
    lam = np.random.beta(alpha, alpha)
    
    # 随机生成一个矩形框 (bbx1, bby1) 和 (bbx2, bby2)，用于融合两张图像的区域
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    
    # 使用另一张图像的相应区域替换第一张图像的相应区域，实现图像融合
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    
    # 在rand_bbox中的np.clip会产生误差，导致裁剪区域比理论上偏少，导致求loss时不准
    # 现基于现实对已给的λ进行一个调整
    # λ = 1 - (融合区域的像素数量 / 总图像像素数量)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, lam]
    return data, targets

def mixup(data, targets1, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam) # 对每个像素点都做融合
    targets = [targets1, shuffled_targets1, lam]

    return data, targets

def cutmix_criterion(preds1, targets):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    # criterion = nn.CrossEntropyLoss(reduction='mean').cuda() # 可被替换成任意loss函数
    # 对cut之外图求loss + cut内图求loss，而面积比 外:内 = lam:(1-lam)
    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2)

def mixup_criterion(preds1, targets):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2) 


# ============================== Function ==============================
def criterion(outputs, targets):
    return nn.BCELoss()(outputs, targets)

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc  = 0.0
    

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        random_number = 0.5
        # random_number = random.random() # 生成一个0到1之间的随机数
        input = images
        tmp = targets
        target = targets
        if random_number < 0.3:
            input,targets=cutmix(input,target,2)
            
            targets[0]=torch.tensor(targets[0]).cuda()
            targets[1]=torch.tensor(targets[1]).cuda()
            targets[2]=torch.tensor(targets[2]).cuda()
        elif random_number > 0.7:
            input,targets=mixup(input,target,2)
            
            targets[0]=torch.tensor(targets[0]).cuda()
            targets[1]=torch.tensor(targets[1]).cuda()
            targets[2]=torch.tensor(targets[2]).cuda()
        else:
            None
        
        
        
        outputs = model(images).squeeze()

        loss=None
        output = outputs
        if random_number < 0.3:
            loss = cutmix_criterion(output, targets) # 注意这是在CPU上运算的
        elif random_number > 0.7:
            loss = mixup_criterion(output, targets) # 注意这是在CPU上运算的
        else:
            loss = criterion(output, target)
        targets = tmp

        # loss = criterion(outputs, targets)
 
 
        loss = loss / CONFIG['n_accumulate']
        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            # if scheduler is not None:
            #     scheduler.step()
                
        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()

        batch_size = images.size(0)
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])
    
    # 修改一下原本代码
    if scheduler is not None:
        scheduler.step()
    gc.collect()
    
    return epoch_loss, epoch_auroc

@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)

        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)

        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()
    
    return epoch_loss, epoch_auroc

def run_training(model, optimizer, scheduler, device, num_epochs):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_auroc = -np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss, train_epoch_auroc = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        
        val_epoch_loss, val_epoch_auroc = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
                                         epoch=epoch)
    
        
        # deep copy the model
        # 新增一个限制,保证val_loss不变大,模型会更稳定
        # if best_epoch_auroc <= val_epoch_auroc and val_epoch_loss<=0.24300:
        if best_epoch_auroc <= val_epoch_auroc:
            print(f"{b_}Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")
            best_epoch_auroc = val_epoch_auroc

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "AUROC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(val_epoch_auroc, val_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUROC: {:.4f}".format(best_epoch_auroc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


def run_test(model, dataloader, device):
    model.eval()
    
    outputs_list = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images).squeeze()

        # 这里要取回到内存，如果不，列表会添加GPU中变量的引用，导致变量不会销毁，最后撑爆GPU
        outputs_list.append(outputs.detach().cpu().numpy())

    
    gc.collect()
    
    return np.concatenate(outputs_list, axis=0)


def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler

def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = ISICDataset_for_Train(df_train, HDF_FILE, transforms=data_transforms["train"])
    train_dataset2020 = ISICDataset_for_Train_fromjpg('/home/xyli/kaggle/data2020', transforms=data_transforms["train"])
    train_dataset2019 = ISICDataset_for_Train_fromjpg('/home/xyli/kaggle/data2019', transforms=data_transforms["train"])
    train_dataset2018 = ISICDataset_for_Train_fromjpg('/home/xyli/kaggle/data2018', transforms=data_transforms["train"])
    train_dataset_others = ISICDataset_for_Train_fromjpg('/home/xyli/kaggle/data_others', transforms=data_transforms["train"])
    # train_dataset_github = ISICDataset_for_Train_github(transforms=data_transforms["train"])
    concat_dataset_train = ConcatDataset([
        train_dataset, train_dataset2020,
        train_dataset2019, train_dataset2018,
        train_dataset_others
    ])

    valid_dataset = ISICDataset(df_valid, HDF_FILE, transforms=data_transforms["valid"])
    valid_dataset2020 = ISICDataset_jpg('/home/xyli/kaggle/data2020', transforms=data_transforms["valid"])
    valid_dataset2019 = ISICDataset_jpg('/home/xyli/kaggle/data2019', transforms=data_transforms["valid"])
    valid_dataset2018 = ISICDataset_jpg('/home/xyli/kaggle/data2018', transforms=data_transforms["valid"])
    valid_dataset_others = ISICDataset_jpg('/home/xyli/kaggle/data_others', transforms=data_transforms["valid"])
    concat_dataset_valid = ConcatDataset([
        valid_dataset, valid_dataset2020,
        valid_dataset2019, valid_dataset2018,
        valid_dataset_others
    ])

    # 用github数据时, num_workers=2
    train_loader = DataLoader(concat_dataset_train, batch_size=CONFIG['train_batch_size'], 
                              num_workers=16, shuffle=True, pin_memory=True, drop_last=True)    
    # train_loader = DataLoader(concat_dataset, batch_size=CONFIG['train_batch_size'], 
    #                           num_workers=2, shuffle=True, pin_memory=True, drop_last=True)

    # from IPython import embed
    # embed()

    valid_loader = DataLoader(concat_dataset_valid, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=16, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader
# ============================== Main ==============================


# train_loader, valid_loader = prepare_loaders(df, fold=CONFIG["fold"])

# optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
#                        weight_decay=CONFIG['weight_decay'])
# scheduler = fetch_scheduler(optimizer)
# model, history = run_training(model, optimizer, scheduler,
#                               device=CONFIG['device'],
#                               num_epochs=CONFIG['epochs'])



# 进行推理
infer_dataset = InferenceDataset( HDF_FILE, transforms=data_transforms["valid"])
test_loader = DataLoader(infer_dataset, 96, num_workers=16, shuffle=False, pin_memory=False)
res = run_test(model, test_loader, device=CONFIG['device']) 

df = pd.read_csv("/home/xyli/kaggle/train-metadata.csv")
df = df[['isic_id', 'target']]

# df = df[0:10000]
df['eva'] = res
df.to_csv('/home/xyli/kaggle/Kaggle_ISIC/eva/eva_train.csv')

# from IPython import embed
# embed()