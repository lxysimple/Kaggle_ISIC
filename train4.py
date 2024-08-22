
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

from sklearn.metrics import hamming_loss, f1_score, roc_curve, auc, classification_report
# ============================== Training Configuration ==============================


CONFIG = {
    "seed": 42,
    
 
    # "img_size": 256,
    # "model_name": "vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k",

    "img_size": 336,
    "model_name": "eva02_small_patch14_336.mim_in22k_ft_in1k",
    
    # "img_size": 384,
    # "model_name": "maxvit_tiny_tf_384",

    # "img_size": 240,
    # "model_name": "tf_efficientnetv2_b3.in21k_ft_in1k",

    # 164: eva、seresnext
    # 64: vit
    "train_batch_size": 164, # 96 32
    
    # 训练时164，
    # eva: 96
    # vit推理: 64
    "valid_batch_size": 164, 


    "scheduler": 'CosineAnnealingLR',
    # "checkpoint": '/home/xyli/kaggle/Kaggle_ISIC/eva/AUROC0.5326_Loss0.2242_pAUC0.1503_fold1.bin',
    # "checkpoint": '/home/xyli/kaggle/Kaggle_ISIC/AUROC0.5318_Loss0.5533_pAUC0.1265_fold1.bin',
    "checkpoint": None,

  
    "learning_rate": 1e-5, # 1e-5
    "min_lr": 1e-6, # 1e-6
    "weight_decay": 1e-6, # 1e-6

    # "learning_rate": 1e-6, # 1e-5
    # "min_lr": 1e-7, # 1e-6
    # "weight_decay": 1e-7, # 1e-6


    # "learning_rate": 1e-4, # 1e-5
    # "min_lr": 1e-5, # 1e-6
    # "weight_decay": 1e-5, # 1e-6

    "T_max": 10,
    "epochs": 10,

    
    "fold" : 0,
    "n_fold": 2,
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


# ------------------------------------- 取比赛原csv
df = pd.read_csv(f"{ROOT_DIR}/train-metadata.csv")
print("        df.shape, # of positive cases, # of patients")
print("original>", df.shape, df.target.sum(), df["patient_id"].unique().shape)
# ===================================== 取比赛原csv


# ------------------------------------- 用聚合算法后的csv
# df2 = pd.read_csv(f"/home/xyli/kaggle/ISIC_2024_Challenge_SelfClean_Scores.csv")

# df = df.merge(df2, on=["isic_id", "patient_id"])

# """
# <102K, 393
# <100K, 393
# """
# index = (df['target'] == 1) & (df['irrelevant_ranking']<90_000)

# df[index]

# from IPython import embed
# embed()
# ===================================== 用聚合算法后的csv


# df_positive = df[df["target"] == 1].reset_index(drop=True) # 取出target=1的所有行
# df_negative = df[df["target"] == 0].reset_index(drop=True) # 取出target=0的所有行
# # 从2个数据集中各自以 positive:negative = 1:20 进行采样，我感觉是确保验证集中正负样本比例为1:10
# # df = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*10, :]])  

# df_train = df_negative.iloc[df_positive.shape[0]*10 : df_positive.shape[0]*10+584*10, :]
# df_valid = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*10, :]])

# print("filtered>", df.shape, df.target.sum(), df["patient_id"].unique().shape)
# df = df.reset_index(drop=True)
# print(df.shape[0], df.target.sum())

# 用于计算一个学习率调整器的一个参数
# 因为之后要合并数据集,算了一下合并后大约是合并前2.4倍,合并前是8k,合并后是20k左右
# CONFIG['T_max'] = 2.4*df.shape[0] * (CONFIG["n_fold"]-1) * CONFIG['epochs'] // CONFIG['train_batch_size'] // CONFIG["n_fold"]
# print(CONFIG['T_max'])





sgkf = StratifiedGroupKFold(n_splits=2)
for fold, ( _, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
      df.loc[val_ , "kfold"] = int(fold)


def show_info(df): 
    """ 统计一下各折的信息  """
    fold_summary = df.groupby("kfold")["patient_id"].nunique().to_dict()
    total_patients = df["patient_id"].nunique()
    print(f"Fold Summary (patients per fold):")
    for fold, count in fold_summary.items():
        if fold != -1:  # Exclude the initialization value
            """ 统计一下数据集总体信息 """
            print(f"Fold {fold}: {count} patients")
            df_flod = df[df['kfold'] == fold]
            print("Original Dataset Summary:")
            print(f"Total number of samples: {len(df_flod)}")
            original_positive_cases = df_flod['target'].sum()
            original_total_cases = len(df_flod)
            original_positive_ratio = original_positive_cases / original_total_cases
            print(f"Number of positive cases: {original_positive_cases}")
            print(f"Number of negative cases: {original_total_cases - original_positive_cases}")
            print(f"Ratio of negative to positive cases: {(original_total_cases - original_positive_cases) / original_positive_cases:.2f}:1")
            print('\n')

    print(f"Total patients: {total_patients}")

""" 
Fold 0.0: 521 patients
Original Dataset Summary:
Total number of samples: 200532
Number of positive cases: 199
Number of negative cases: 200333
Ratio of negative to positive cases: 1006.70:1

Fold 1.0: 521 patients
Original Dataset Summary:
Total number of samples: 200527
Number of positive cases: 194
Number of negative cases: 200333
Ratio of negative to positive cases: 1032.64:1
"""
show_info(df)


# ------------------------------------- 对各折下采样
tmp_sum = pd.DataFrame()
for i in range(2):
    df_fold = df[df['kfold'] == i]
    df_positive = df_fold[df_fold["target"] == 1].reset_index(drop=True) # 取出target=1的所有行
    df_negative = df_fold[df_fold["target"] == 0].reset_index(drop=True) # 取出target=0的所有行
    # 从2个数据集中各自以 positive:negative = 1:20 进行采样，我感觉是确保验证集中正负样本比例为1:10
    # tmp = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*10, :]]) 

    if CONFIG['fold'] != i:
        positive_list = []
        for i in range(1):
            positive_list.append(df_positive)
            # continue
        positive_list.append(df_negative.iloc[:df_positive.shape[0]*10, :]) 
        # positive_list.append(df_negative) 
        tmp = pd.concat(positive_list) 
    else:
        tmp = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*10, :]]) 

    tmp_sum = pd.concat([tmp_sum, tmp]) 

df = tmp_sum
 
"""
Total patients: 1042
Fold Summary (patients per fold):
Fold 0.0: 428 patients
Original Dataset Summary:
Total number of samples: 2189
Number of positive cases: 199
Number of negative cases: 1990
Ratio of negative to positive cases: 10.00:1


Fold 1.0: 432 patients
Original Dataset Summary:
Total number of samples: 2134
Number of positive cases: 194
Number of negative cases: 1940
Ratio of negative to positive cases: 10.00:1
"""
show_info(df)
# ========================================== 对各折下采样


# from IPython import embed
# embed()
# ============================== Dataset Class ==============================

class ISICDataset(Dataset):
    def __init__(self, df, file_hdf, transforms=None):
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.df = df
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array( Image.open(BytesIO(self.fp_hdf[isic_id][()])) )

        if self.targets is not None:
            target = self.targets[index]
        else:
            target = torch.tensor(-1)  # Dummy target for test set
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
        
        
        return {
            'image': img,
            'target': target
        }

class InferenceDataset(Dataset):
    def __init__(self, file_hdf, transforms=None):
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.df = pd.read_csv("/home/xyli/kaggle/train-metadata.csv")
        # self.df = self.df[0:10000]
        self.isic_ids = self.df['isic_id'].values
        # self.targets = df['target'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array( Image.open(BytesIO(self.fp_hdf[isic_id][()])) )
        # target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img
        }


class ISICDataset_for_Train_fromjpg(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        df = pd.read_csv(f"{path}/train-metadata.csv")

        # df_2024 = pd.read_csv(f"{ROOT_DIR}/train-metadata.csv")
        # self.df_negative = df_2024[df_2024["target"] == 0].reset_index()
        # self.pic_2024 = h5py.File(HDF_FILE, mode="r")

        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()
        # 保持一定的正负比例，不能让其失衡
        # start = CONFIG['fold']*len(self.df_positive)*10
        start = len(self.df_positive)*10
        # start = 0
        self.df_negative = self.df_negative[0 : start]

        self.df = pd.concat([self.df_positive, self.df_negative]) 
        # self.df = pd.concat([self.df_positive, self.df_positive, self.df_negative]) 
        # self.df = self.df_positive
        self.isic_ids = self.df['isic_id'].values
        self.targets = self.df['target'].values

        self.transforms = transforms

        print(path)
        print(len(self.df_positive), ' ', len(self.df_negative))

        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        isic_id = self.isic_ids[index]
        img = np.array( Image.open(f"{self.path}/train-image/image/{isic_id}.jpg") )
        target = self.targets[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }

# ============================== Create Model ==============================

class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(ISICModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.sigmoid = nn.Sigmoid()

        # in_features = self.model.head.in_features
        # self.model.head = nn.Linear(in_features, num_classes)

        self.model.reset_classifier(num_classes=num_classes)
        
    def forward(self, images):
        return self.sigmoid(self.model(images))
    



# class ISICModel(nn.Module):
#     def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
#         super(ISICModel, self).__init__()
#         self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

#         in_features = self.model.head.in_features # eva
#         # in_features = 1000 # vit

#         # self.model.head = nn.Linear(in_features, num_classes)

#         self.head = nn.Linear(in_features, num_classes)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, images):
#         x = self.model(images)
#         x = self.head(x)
#         # print('res.shape: ', x.shape)

#         return self.sigmoid(x)



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
data_transforms = {
    "train": A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()
    ], p=1.),

    # "train":A.Compose([
        
    #         A.Transpose(p=0.5),
    #         A.VerticalFlip(p=0.5),
    #         A.HorizontalFlip(p=0.5),
    #         A.ColorJitter(brightness=0.2, p=0.75), # A.RandomBrightness(limit=0.2, p=0.75),
    #         A.ColorJitter(contrast=0.2, p=0.75), # A.RandomContrast(limit=0.2, p=0.75),
    #         # A.OneOf([
    #         #     A.MotionBlur(blur_limit=5),
    #         #     A.MedianBlur(blur_limit=5),
    #         #     A.GaussianBlur(blur_limit=5),
    #         #     A.GaussNoise(var_limit=(5.0, 30.0)),
    #         # ], p=0.7),
    #         A.MotionBlur(blur_limit=5, p=0.7),
    #         A.MedianBlur(blur_limit=5, p=0.7),
    #         A.GaussianBlur(blur_limit=5, p=0.7),
    #         A.GaussNoise(var_limit=(5.0, 30.0), p=0.7),

    #         # A.OneOf([
    #         #     A.OpticalDistortion(distort_limit=1.0),
    #         #     A.GridDistortion(num_steps=5, distort_limit=1.),
    #         #     A.ElasticTransform(alpha=3),
    #         # ], p=0.7),
    #         A.OpticalDistortion(distort_limit=1.0, p=0.7),
    #         A.GridDistortion(num_steps=5, distort_limit=1.0, p=0.7),
    #         A.ElasticTransform(alpha=3, p=0.7),

    #         A.CLAHE(clip_limit=4.0, p=0.7),
    #         A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    #         A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),

    #         A.Resize(CONFIG['img_size'], CONFIG['img_size']),
    #         # A.Cutout(max_h_size=int(CONFIG['img_size'] * 0.375), max_w_size=int(CONFIG['img_size'] * 0.375), num_holes=1, p=0.7), 
    #         A.CoarseDropout(p=0.7), # == Cutout
    #         A.Normalize(
    #                 mean=[0.4815, 0.4578, 0.4082], 
    #                 std=[0.2686, 0.2613, 0.2758], 
    #                 max_pixel_value=255.0,
    #                 p=1.0
    #             ),
    #         ToTensorV2()
    # ], p=1.),

    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2(),
        ], p=1.)
    

}

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
        if random_number < -1:
            input,targets=cutmix(input,target,2)
            
            targets[0]=torch.tensor(targets[0]).cuda()
            targets[1]=torch.tensor(targets[1]).cuda()
            targets[2]=torch.tensor(targets[2]).cuda()
        elif random_number > 1:
            input,targets=mixup(input,target,0.5)
            
            targets[0]=torch.tensor(targets[0]).cuda()
            targets[1]=torch.tensor(targets[1]).cuda()
            targets[2]=torch.tensor(targets[2]).cuda()
        else:
            None
        
        
        
        outputs = model(images).squeeze()

        loss=None
        output = outputs
        if random_number < -1:
            loss = cutmix_criterion(output, targets) # 注意这是在CPU上运算的
        elif random_number > 1:
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
    
    val_targets, val_outputs = [], []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)

        outputs = model(images).squeeze()

        val_targets.append(targets.cpu())
        val_outputs.append(outputs.cpu())

        loss = criterion(outputs, targets)



        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)


        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        try:
            bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])   
        except:
            continue
    
    gc.collect()
    
    return epoch_loss, epoch_auroc, torch.cat(val_targets).numpy(), torch.cat(val_outputs).numpy()

def run_training(model, optimizer, scheduler, device, num_epochs):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_auroc = -np.inf
    best_pauc = 0
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss, train_epoch_auroc = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        
        val_epoch_loss, val_epoch_auroc,\
        epoch_val_targets, epoch_val_outputs = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
                                         epoch=epoch)
        
        # Create DataFrames with row_id for scoring
        solution_df = pd.DataFrame({'target': epoch_val_targets, 'row_id': range(len(epoch_val_targets))})
        submission_df = pd.DataFrame({'prediction': epoch_val_outputs, 'row_id': range(len(epoch_val_outputs))})
        epoch_score = score(solution_df, submission_df, 'row_id')
        print("epoch_score: {:.4f}".format(epoch_score))

        # deep copy the model
        # 新增一个限制,保证val_loss不变大,模型会更稳定
        # if best_epoch_auroc <= val_epoch_auroc and val_epoch_loss<=0.24300:
        if best_pauc <= epoch_score:
            print(f"{b_}pAUROC Improved ({best_pauc} ---> {epoch_score})")
            best_pauc = epoch_score

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "AUROC{:.4f}_Loss{:.4f}_pAUC{:.4f}_fold{:.0f}.bin".\
                format(val_epoch_auroc, val_epoch_loss, best_pauc, CONFIG['fold'])

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
    
    train_dataset = ISICDataset(df_train, HDF_FILE, transforms=data_transforms["train"])
    train_dataset2020 = ISICDataset_for_Train_fromjpg('/home/xyli/kaggle/data2020', transforms=data_transforms["train"])
    train_dataset2019 = ISICDataset_for_Train_fromjpg('/home/xyli/kaggle/data2019', transforms=data_transforms["train"])
    train_dataset2018 = ISICDataset_for_Train_fromjpg('/home/xyli/kaggle/data2018', transforms=data_transforms["train"])
    train_dataset_others = ISICDataset_for_Train_fromjpg('/home/xyli/kaggle/data_others', transforms=data_transforms["train"])
     

    valid_dataset = ISICDataset(df_valid, HDF_FILE, transforms=data_transforms["valid"])

    concat_dataset_train = ConcatDataset([
        train_dataset2020, 
        # train_dataset2018,
        train_dataset, 
        # train_dataset2019,
        train_dataset_others,

    ])

    # 用github数据时, num_workers=2
    train_loader = DataLoader(concat_dataset_train, batch_size=CONFIG['train_batch_size'], 
                              num_workers=16, shuffle=True, pin_memory=True, drop_last=True)    
    # train_loader = DataLoader(concat_dataset, batch_size=CONFIG['train_batch_size'], 
    #                           num_workers=2, shuffle=True, pin_memory=True, drop_last=True)


    # from IPython import embed
    # embed()

    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=16, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader
# ============================== Main ==============================


# ------------------------------------------------------------------ 模型训练
train_loader, valid_loader = prepare_loaders(df, CONFIG['fold'])

optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                       weight_decay=CONFIG['weight_decay'])
scheduler = fetch_scheduler(optimizer)
model, history = run_training(model, optimizer, scheduler,
                              device=CONFIG['device'],
                              num_epochs=CONFIG['epochs'])
# ================================================================== 模型训练


# ------------------------------------------------------------------ 进行推理
# def load_model(path):
#     model = ISICModel(CONFIG['model_name'], pretrained=False)
#     checkpoint = torch.load(path)
#     print(f"load checkpoint: {path}") 
#     # 去掉前面多余的'module.'
#     new_state_dict = {}
#     for k,v in checkpoint.items():
#         new_state_dict[k[7:]] = v
#     model.load_state_dict( new_state_dict )

#     model = model.cuda() 
#     # model.to(CONFIG['device'])
#     model = DataParallel(model) 
#     return model

# models = []
# models.append(load_model('/home/xyli/kaggle/Kaggle_ISIC/eva/AUROC0.5341_Loss0.1620_pAUC0.1523_fold0.bin'))
# models.append(load_model('/home/xyli/kaggle/Kaggle_ISIC/eva/AUROC0.5323_Loss0.1705_pAUC0.1419_fold1.bin'))

# df = pd.read_csv("/home/xyli/kaggle/train-metadata.csv")
# sgkf = StratifiedGroupKFold(n_splits=2)
# for fold, ( _, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
#       df.loc[val_ , "kfold"] = int(fold)

# df_valids = pd.DataFrame()
# for i in range(CONFIG['n_fold']):
#     _, valid_loader = prepare_loaders(df, i)
#     res = run_test(models[i], valid_loader, device=CONFIG['device']) 
#     df_valid = df[df.kfold == i].reset_index()
#     df_valid['eva'] = res
#     df_valids = pd.concat([df_valids, df_valid])

# # from IPython import embed
# # embed()

# df_valids = df_valids[["isic_id", "patient_id", "eva"]]


# df = df[['isic_id', 'patient_id', 'target']]
# df = df.merge(df_valids, on=["isic_id", "patient_id"])


# try:
#     df = df[['isic_id', 'patient_id', 'target', "eva"]]
#     df.to_csv('/home/xyli/kaggle/Kaggle_ISIC/eva/eva_train2.csv')
# except:

#     df.rename(columns={'target_x': 'target'}, inplace=True)
#     df = df[['isic_id', 'patient_id', 'target', "eva"]]
#     df.to_csv('/home/xyli/kaggle/Kaggle_ISIC/eva/eva_train2.csv')

# ===================================================================== 进行推理

# --------------------------------------------------------------------- 测试BUG
# def load_model(path):
#     model = ISICModel(CONFIG['model_name'], pretrained=False)
#     checkpoint = torch.load(path)
#     print(f"load checkpoint: {path}") 
#     # 去掉前面多余的'module.'
#     new_state_dict = {}
#     for k,v in checkpoint.items():
#         new_state_dict[k[7:]] = v
#     model.load_state_dict( new_state_dict )
#     model = model.cuda() 
#     return model


# model = load_model('/home/xyli/kaggle/Kaggle_ISIC/eva/AUROC0.5336_Loss0.2118_pAUC0.1514_fold0.bin')


# _, valid_loader = prepare_loaders(df, 0)

# _, _, epoch_val_targets, epoch_val_outputs = valid_one_epoch(
#             model, valid_loader, device=CONFIG['device'], epoch=999)

# # Create DataFrames with row_id for scoring
# solution_df = pd.DataFrame({'target': epoch_val_targets, 'row_id': range(len(epoch_val_targets))})
# submission_df = pd.DataFrame({'prediction': epoch_val_outputs, 'row_id': range(len(epoch_val_outputs))})
# epoch_score = score(solution_df, submission_df, 'row_id')
# print("epoch_score: {:.4f}".format(epoch_score))
# ===================================================================== 测试BUG