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
def setup_model(num_classes=2, freeze_base_model=freeze_base_model):
    model = timm.create_model('tf_efficientnetv2_b1', 
                            checkpoint_path='/kaggle/input/effnetv2-m-b1-pth/tf_efficientnetv2_b1-be6e41b0.pth',
                            pretrained=False)

    if freeze_base_model:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = nn.Linear(model.classifier.in_features, out_features=num_classes)
    return model.to(device)    

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



# from IPython import embed
# embed()
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
def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80) -> float:

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray(solution.values)-1)
    
    # flip the submissions to their compliments
    v_pred = -1.0*np.asarray(submission.values)

    max_fpr = abs(1-min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
        
    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    return(partial_auc)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np

def train_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, fold, epoch, device):
    scaler = GradScaler()
    
    # Training phase
    model.train()
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}, Fold {fold+1} Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # Evaluation phase
    model.eval()
    val_targets, val_outputs = [], []
    with torch.no_grad(), autocast():
        for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}, Fold {fold+1} Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_targets.append(targets.cpu())
            val_outputs.append(outputs.softmax(dim=1)[:, 1].cpu())
    
    scheduler.step()
    return torch.cat(val_targets).numpy(), torch.cat(val_outputs).numpy()


def cross_validation_train(df_train, num_folds, num_epochs, hdf5_file_path, aug_transform, base_transform, device):
    criterion = nn.CrossEntropyLoss()
    all_val_targets, all_val_outputs = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_val_targets, epoch_val_outputs = [], []

        for fold in range(num_folds):
            print(f"\nFold {fold + 1}/{num_folds}")
            
            # Split data for current fold
            train_df = df_train[df_train['fold'] != fold]
            val_df = df_train[df_train['fold'] == fold]
            
            # Create datasets and data loaders
            train_dataset = ISICDataset(hdf5_file_path, train_df['isic_id'].values, train_df['target'].values, aug_transform)
            val_dataset = ISICDataset(hdf5_file_path, val_df['isic_id'].values, val_df['target'].values, base_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
            
            # Initialize model, optimizer, and scheduler
            model = setup_model().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            
            print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
                  f"Train Pos Ratio: {train_df['target'].mean():.2%}, Val Pos Ratio: {val_df['target'].mean():.2%}")
            
            # Train and evaluate
            val_targets, val_outputs = train_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, fold, epoch, device)
            epoch_val_targets.extend(val_targets)
            epoch_val_outputs.extend(val_outputs)
            
            torch.save(model.state_dict(), f'model_fold_{fold}_epoch_{epoch + 1}.pth')
            
            # Create DataFrames with row_id for scoring
            solution_df = pd.DataFrame({'target': val_targets, 'row_id': range(len(val_targets))})
            submission_df = pd.DataFrame({'prediction': val_outputs, 'row_id': range(len(val_outputs))})
            fold_score = score(solution_df, submission_df, 'row_id')
            print(f'Fold {fold + 1} pAUC Score: {fold_score:.4f}')
        
        all_val_targets.extend(epoch_val_targets)
        all_val_outputs.extend(epoch_val_outputs)
        
        # Create DataFrames with row_id for scoring
        solution_df = pd.DataFrame({'target': epoch_val_targets, 'row_id': range(len(epoch_val_targets))})
        submission_df = pd.DataFrame({'prediction': epoch_val_outputs, 'row_id': range(len(epoch_val_outputs))})
        cv_score = score(solution_df, submission_df, 'row_id')
        print(f'Epoch {epoch + 1}/{num_epochs} CV pAUC Score: {cv_score:.4f}')

    return np.array(all_val_targets), np.array(all_val_outputs)
# ============================== Main ==============================

# Set up CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Perform cross-validation training
all_val_targets, all_val_outputs = cross_validation_train(df_train_balanced, 5, 20, HDF_FILE, aug_transform, base_transform, device)






