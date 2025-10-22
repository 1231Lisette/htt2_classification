#!/usr/bin/env python3
"""
数据集检查脚本
验证标签文件、类别映射、数据完整性
"""

import os
import glob
import argparse
from collections import Counter, defaultdict

# 类别映射（根据题目说明）
CLASS_MAPPING = {
    0: "控制适应标识",
    1: "无螺丝", 
    2: "有螺丝",
    3: "电路板"
}

def check_yolo_dataset(data_dir):
    """
    检查YOLO格式数据集的完整性
    
    Args:
        data_dir: 数据集目录（包含images和labels子目录）
    """
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    
    print(f"检查数据集: {data_dir}")
    print("=" * 50)
    
    # 检查目录是否存在
    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在: {images_dir}")
        return False
    
    if not os.path.exists(labels_dir):
        print(f"错误: 标签目录不存在: {labels_dir}")
        return False
    
    # 获取文件列表
    image_files = set([os.path.splitext(f)[0] for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    label_files = set([os.path.splitext(f)[0] for f in os.listdir(labels_dir) 
                      if f.endswith('.txt')])
    
    print(f"图像文件数量: {len(image_files)}")
    print(f"标签文件数量: {len(label_files)}")
    
    # 检查文件对应关系
    missing_images = label_files - image_files
    missing_labels = image_files - label_files
    common_files = image_files & label_files
    
    if missing_images:
        print(f"警告: {len(missing_images)} 个标签文件没有对应的图像文件:")
        for f in list(missing_images)[:5]:  # 只显示前5个
            print(f"  - {f}")
        if len(missing_images) > 5:
            print(f"  ... 还有 {len(missing_images) - 5} 个")
    
    if missing_labels:
        print(f"警告: {len(missing_labels)} 个图像文件没有对应的标签文件:")
        for f in list(missing_labels)[:5]:
            print(f"  - {f}")
        if len(missing_labels) > 5:
            print(f"  ... 还有 {len(missing_labels) - 5} 个")
    
    print(f"有效文件对: {len(common_files)}")
    
    # 检查标签文件内容
    print("\n检查标签文件内容...")
    class_distribution = Counter()
    bbox_stats = defaultdict(list)
    error_files = []
    
    for label_file in common_files:
        label_path = os.path.join(labels_dir, label_file + '.txt')
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"错误: {label_file}.txt 第{line_num}行格式不正确: {line}")
                    error_files.append(label_file)
                    continue
                
                try:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 检查类别ID是否有效
                    if cls_id not in CLASS_MAPPING:
                        print(f"错误: {label_file}.txt 第{line_num}行 - 无效类别ID: {cls_id}")
                        error_files.append(label_file)
                        continue
                    
                    # 检查坐标是否在有效范围内
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 <= width <= 1 and 0 <= height <= 1):
                        print(f"错误: {label_file}.txt 第{line_num}行 - 坐标超出范围: {x_center}, {y_center}, {width}, {height}")
                        error_files.append(label_file)
                        continue
                    
                    # 统计类别分布
                    class_distribution[cls_id] += 1
                    
                    # 统计边界框尺寸
                    bbox_stats[cls_id].append((width, height))
                    
                except ValueError as e:
                    print(f"错误: {label_file}.txt 第{line_num}行 - 数值格式错误: {e}")
                    error_files.append(label_file)
                    
        except Exception as e:
            print(f"错误: 读取文件 {label_file}.txt 时出错: {e}")
            error_files.append(label_file)
    
    # 输出类别分布
    print("\n类别分布统计:")
    print("-" * 30)
    total_objects = sum(class_distribution.values())
    for cls_id in sorted(class_distribution.keys()):
        count = class_distribution[cls_id]
        percentage = (count / total_objects) * 100
        class_name = CLASS_MAPPING.get(cls_id, "未知")
        print(f"  {cls_id}: {class_name} - {count} 个 ({percentage:.1f}%)")
    
    # 输出边界框统计
    print("\n边界框尺寸统计:")
    print("-" * 30)
    for cls_id in sorted(bbox_stats.keys()):
        widths = [w for w, h in bbox_stats[cls_id]]
        heights = [h for w, h in bbox_stats[cls_id]]
        
        if widths and heights:
            avg_width = sum(widths) / len(widths)
            avg_height = sum(heights) / len(heights)
            class_name = CLASS_MAPPING.get(cls_id, "未知")
            print(f"  {cls_id}: {class_name}")
            print(f"    平均尺寸: {avg_width:.3f} x {avg_height:.3f}")
            print(f"    最小尺寸: {min(widths):.3f} x {min(heights):.3f}")
            print(f"    最大尺寸: {max(widths):.3f} x {max(heights):.3f}")
    
    # 总结
    print("\n检查总结:")
    print("-" * 30)
    if not missing_images and not missing_labels and not error_files:
        print("✅ 数据集检查通过!")
        return True
    else:
        print("❌ 数据集存在问题:")
        if missing_images:
            print(f"  - {len(missing_images)} 个标签文件缺少对应图像")
        if missing_labels:
            print(f"  - {len(missing_labels)} 个图像文件缺少对应标签")
        if error_files:
            print(f"  - {len(set(error_files))} 个标签文件存在格式错误")
        return False

def main():
    parser = argparse.ArgumentParser(description='YOLO数据集检查工具')
    parser.add_argument('--data_dir', required=True, help='YOLO数据集目录')
    
    args = parser.parse_args()
    
    success = check_yolo_dataset(args.data_dir)
    
    # 退出码：0表示成功，1表示失败
    exit(0 if success else 1)

if __name__ == '__main__':
    main()