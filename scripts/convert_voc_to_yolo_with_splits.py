#!/usr/bin/env python3
"""
VOC格式转YOLO格式转换器 - 最终版本，只使用train和val
"""

import os
import xml.etree.ElementTree as ET
import argparse
import shutil

# 类别映射（修正后的）
CLASS_MAPPING = {
    0: "控制适应标识",
    1: "无螺丝", 
    2: "有螺丝",
    3: "电路板"
}

def convert_voc_to_yolo_final(voc_annotations_dir, voc_images_dir, imagesets_dir, output_dir):
    """
    将VOC数据集转换为YOLO格式，只使用train和val划分
    """
    # 创建输出目录
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    print(f"开始VOC到YOLO格式转换（仅使用train和val）...")
    print(f"类别映射: {CLASS_MAPPING}")
    
    # 读取划分文件
    train_split_file = os.path.join(imagesets_dir, 'Main', 'train.txt')
    val_split_file = os.path.join(imagesets_dir, 'Main', 'val.txt')
    
    if not os.path.exists(train_split_file):
        raise FileNotFoundError(f"训练集划分文件不存在: {train_split_file}")
    if not os.path.exists(val_split_file):
        raise FileNotFoundError(f"验证集划分文件不存在: {val_split_file}")
    
    # 读取训练集文件列表
    with open(train_split_file, 'r', encoding='utf-8') as f:
        train_files = [line.strip() for line in f.readlines()]
    
    # 读取验证集文件列表
    with open(val_split_file, 'r', encoding='utf-8') as f:
        val_files = [line.strip() for line in f.readlines()]
    
    print(f"训练集样本数: {len(train_files)}")
    print(f"验证集样本数: {len(val_files)}")
    print("注意: 不使用测试集，按照老师的原始划分")
    
    # 处理训练集
    print("\n处理训练集...")
    processed_train = process_split(voc_annotations_dir, voc_images_dir, 
                                   train_images_dir, train_labels_dir, train_files)
    
    # 处理验证集
    print("\n处理验证集...")
    processed_val = process_split(voc_annotations_dir, voc_images_dir, 
                                 val_images_dir, val_labels_dir, val_files)
    
    print(f"\n转换完成!")
    print(f"训练集: {processed_train} 个文件")
    print(f"验证集: {processed_val} 个文件")
    print(f"总计: {processed_train + processed_val} 个文件")
    print(f"输出目录: {output_dir}")

def process_split(voc_annotations_dir, voc_images_dir, output_images_dir, output_labels_dir, file_list):
    """处理单个划分"""
    processed_count = 0
    
    for file_base in file_list:
        xml_path = os.path.join(voc_annotations_dir, file_base + '.xml')
        
        if not os.path.exists(xml_path):
            print(f"警告: XML文件不存在: {xml_path}")
            continue
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取图像信息
            size = root.find('size')
            if size is None:
                print(f"警告: {file_base}.xml 缺少尺寸信息，跳过")
                continue
                
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            # 获取图像文件名
            image_name = root.find('filename').text
            if image_name is None:
                print(f"警告: {file_base}.xml 缺少文件名，跳过")
                continue
                
            image_path = os.path.join(voc_images_dir, image_name)
            
            # 创建YOLO标签文件
            txt_filename = file_base + '.txt'
            txt_path = os.path.join(output_labels_dir, txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                for obj in root.iter('object'):
                    cls_name = obj.find('name').text
                    
                    # 查找类别ID
                    cls_id = None
                    for id_val, name_val in CLASS_MAPPING.items():
                        if name_val == cls_name:
                            cls_id = id_val
                            break
                    
                    if cls_id is None:
                        print(f"警告: 未知类别 '{cls_name}'，跳过")
                        continue
                    
                    bbox = obj.find('bndbox')
                    if bbox is None:
                        print(f"警告: {file_base}.xml 中对象缺少边界框，跳过")
                        continue
                        
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # 转换为YOLO格式
                    x_center = (xmin + xmax) / 2.0 / img_width
                    y_center = (ymin + ymax) / 2.0 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    # 确保坐标在有效范围内
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # 复制图像文件
            if os.path.exists(image_path):
                shutil.copy2(image_path, output_images_dir)
                processed_count += 1
            else:
                print(f"警告: 图像文件不存在 {image_path}")
                
        except Exception as e:
            print(f"处理文件 {file_base} 时出错: {e}")
    
    return processed_count

def main():
    parser = argparse.ArgumentParser(description='VOC转YOLO格式转换器（仅train和val）')
    parser.add_argument('--voc_annotations', required=True, help='VOC标注文件目录')
    parser.add_argument('--voc_images', required=True, help='VOC图像文件目录')
    parser.add_argument('--imagesets', required=True, help='ImageSets目录')
    parser.add_argument('--output', required=True, help='YOLO格式输出目录')
    
    args = parser.parse_args()
    
    convert_voc_to_yolo_final(args.voc_annotations, args.voc_images, 
                             args.imagesets, args.output)

if __name__ == '__main__':
    main()