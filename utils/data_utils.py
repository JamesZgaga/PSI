import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FaceDataset(Dataset):
    def __init__(self, df, img_size=224, augment=False, crop_face=True):
        self.df = df
        self.img_size = img_size
        self.crop_face = crop_face
        
        # 定义数据增强
        if augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 如果需要裁剪人脸区域且有边界框信息
        if self.crop_face and 'bbox' in row:
            x1, y1, x2, y2 = eval(row['bbox']) if isinstance(row['bbox'], str) else row['bbox']
            # 扩大边界框以包含更多上下文
            h, w = img.shape[:2]
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            img = img[y1:y2, x1:x2]
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        # 获取标签
        label = torch.tensor(row['class_id'], dtype=torch.long)
        
        return img, label

def create_dataloaders(train_df, val_df, img_size=224, batch_size=32, num_workers=4):
    """创建训练和验证数据加载器"""
    train_dataset = FaceDataset(train_df, img_size=img_size, augment=True)
    val_dataset = FaceDataset(val_df, img_size=img_size, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
