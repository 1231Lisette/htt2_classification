"""
数据加载和预处理模块
整合了数据集定义和数据增强变换
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HT22Dataset(Dataset):
    """HT22控制框检测数据集类"""
    
    def __init__(self, data_dir, img_size=640, augment=False, mode='train'):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            img_size: 图像尺寸
            augment: 是否进行数据增强
            mode: 模式 ('train'/'val')
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        self.mode = mode
        self.image_files = []
        self.label_files = []
        
        # 加载数据路径
        self._load_data_paths()
        
        # 数据增强变换
        self.transform = self._get_transforms()
    
    def _load_data_paths(self):
        """加载图像和标签文件路径"""
        images_dir = os.path.join(self.data_dir, 'images')
        labels_dir = os.path.join(self.data_dir, 'labels')
        
        # 检查目录是否存在
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"图像目录不存在: {images_dir}")
        
        # 遍历图像文件
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(images_dir, img_file)
                label_path = os.path.join(labels_dir, 
                                        os.path.splitext(img_file)[0] + '.txt')
                
                # 只有标签文件存在时才加入
                if os.path.exists(label_path):
                    self.image_files.append(img_path)
                    self.label_files.append(label_path)
        
        print(f"加载 {self.mode} 数据集: {len(self.image_files)} 个样本")
    
    def _get_transforms(self):
        """获取数据变换管道"""
        if self.augment and self.mode == 'train':
            # 训练时的数据增强
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            # 验证/测试时的基本变换
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            image: 预处理后的图像张量
            target: 标注信息字典
            img_path: 图像路径（用于调试）
        """
        # 读取图像
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标签
        labels = self._load_labels(self.label_files[idx])
        
        if len(labels) > 0:
            # 提取边界框和类别
            bboxes = labels[:, 1:5]
            class_labels = labels[:, 0]
            
            # 应用变换
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            image = transformed['image']
            bboxes = np.array(transformed['bboxes'])
            class_labels = np.array(transformed['class_labels'])
            
            # 重新组合标签
            if len(bboxes) > 0:
                labels = np.column_stack([class_labels, bboxes])
            else:
                labels = np.zeros((0, 5))
        else:
            # 无目标的情况
            transformed = self.transform(image=image)
            image = transformed['image']
            labels = np.zeros((0, 5))
        
        # 转换为模型需要的格式
        target = {
            'labels': torch.from_numpy(labels).float(),
            'img_path': img_path
        }
        
        return image, target, img_path
    
    def _load_labels(self, label_path):
        """加载YOLO格式标签文件"""
        labels = []
        try:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:  # 确保每行有5个值
                        class_id, x_center, y_center, width, height = map(float, data)
                        labels.append([class_id, x_center, y_center, width, height])
        except FileNotFoundError:
            print(f"警告: 标签文件不存在 {label_path}")
        
        return np.array(labels) if labels else np.zeros((0, 5))