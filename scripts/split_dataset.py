#!/usr/bin/env python3
"""
数据集划分脚本
将完整的YOLO格式数据集划分为训练集、验证集、测试集
"""

import os
import random
import shutil
from sklearn.model_selection import train_test_split
import argparse  

def split_yolo_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    划分YOLO格式数据集
    
    Args:
        input_dir: 输入数据集目录（包含images和labels）
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"比例总和必须为1.0，当前为: {total_ratio}")
    
    input_images_dir = os.path.join(input_dir, 'images')
    input_labels_dir = os.path.join(input_dir, 'labels')
    
    # 获取所有图像文件（不含扩展名）
    image_files = [os.path.splitext(f)[0] for f in os.listdir(input_images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"总样本数: {len(image_files)}")
    print(f"划分比例: 训练集 {train_ratio*100}%, 验证集 {val_ratio*100}%, 测试集 {test_ratio*100}%")
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 划分数据集
    train_files, temp_files = train_test_split(
        image_files, test_size=(val_ratio + test_ratio), random_state=42
    )
    
    # 计算验证集和测试集的比例
    val_test_ratio = val_ratio + test_ratio
    val_ratio_adjusted = val_ratio / val_test_ratio
    test_ratio_adjusted = test_ratio / val_test_ratio
    
    val_files, test_files = train_test_split(
        temp_files, test_size=test_ratio_adjusted, random_state=42
    )
    
    print(f"训练集: {len(train_files)} 个样本")
    print(f"验证集: {len(val_files)} 个样本")
    print(f"测试集: {len(test_files)} 个样本")
    
    # 创建输出目录
    splits = ['train', 'val', 'test']
    split_files = [train_files, val_files, test_files]
    
    for split, files in zip(splits, split_files):
        split_images_dir = os.path.join(output_dir, split, 'images')
        split_labels_dir = os.path.join(output_dir, split, 'labels')
        
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)
        
        # 复制文件
        for file_base in files:
            # 查找原始图像文件（支持多种格式）
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                src_image = os.path.join(input_images_dir, file_base + ext)
                if os.path.exists(src_image):
                    dst_image = os.path.join(split_images_dir, file_base + '.jpg')
                    shutil.copy2(src_image, dst_image)
                    break
            
            # 复制标签文件
            src_label = os.path.join(input_labels_dir, file_base + '.txt')
            dst_label = os.path.join(split_labels_dir, file_base + '.txt')
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告: 标签文件不存在: {src_label}")
    
    print(f"\n数据集划分完成!")
    print(f"输出目录: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='YOLO数据集划分工具')
    parser.add_argument('--input_dir', required=True, help='输入YOLO数据集目录')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='测试集比例')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    if not os.path.exists(os.path.join(args.input_dir, 'images')):
        print(f"错误: 输入目录缺少images子目录: {args.input_dir}")
        return
    
    if not os.path.exists(os.path.join(args.input_dir, 'labels')):
        print(f"错误: 输入目录缺少labels子目录: {args.input_dir}")
        return
    
    split_yolo_dataset(args.input_dir, args.output_dir, 
                      args.train_ratio, args.val_ratio, args.test_ratio)

if __name__ == '__main__':
    main()