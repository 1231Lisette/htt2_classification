#!/usr/bin/env python3
"""
检查VOC数据集中的类别分布和标注质量
"""

import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import argparse

def check_voc_dataset(voc_annotations_dir, voc_images_dir):
    """
    检查VOC数据集的完整性和质量
    
    Args:
        voc_annotations_dir: VOC标注文件目录
        voc_images_dir: VOC图像文件目录
    """
    print("检查VOC数据集...")
    print("=" * 50)
    
    # 检查目录是否存在
    if not os.path.exists(voc_annotations_dir):
        print(f"错误: VOC标注目录不存在: {voc_annotations_dir}")
        return False
    
    if not os.path.exists(voc_images_dir):
        print(f"错误: VOC图像目录不存在: {voc_images_dir}")
        return False
    
    # 统计信息
    xml_files = [f for f in os.listdir(voc_annotations_dir) if f.endswith('.xml')]
    image_files = [f for f in os.listdir(voc_images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"XML标注文件数量: {len(xml_files)}")
    print(f"图像文件数量: {len(image_files)}")
    
    # 文件对应关系检查
    xml_basenames = set([os.path.splitext(f)[0] for f in xml_files])
    image_basenames = set([os.path.splitext(f)[0] for f in image_files])
    
    missing_images = xml_basenames - image_basenames
    missing_xmls = image_basenames - xml_basenames
    
    if missing_images:
        print(f"警告: {len(missing_images)} 个XML文件没有对应的图像:")
        for f in list(missing_images)[:3]:
            print(f"  - {f}.xml")
        if len(missing_images) > 3:
            print(f"  ... 还有 {len(missing_images)-3} 个")
    
    if missing_xmls:
        print(f"警告: {len(missing_xmls)} 个图像文件没有对应的XML标注:")
        for f in list(missing_xmls)[:3]:
            print(f"  - {f}")
        if len(missing_xmls) > 3:
            print(f"  ... 还有 {len(missing_xmls)-3} 个")
    
    # 检查XML文件内容
    print("\n检查XML文件内容...")
    class_distribution = Counter()
    bbox_stats = defaultdict(list)
    image_sizes = []
    error_files = []
    
    for xml_file in xml_files:
        xml_path = os.path.join(voc_annotations_dir, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 检查图像尺寸
            size = root.find('size')
            if size is None:
                print(f"警告: {xml_file} 缺少尺寸信息")
                error_files.append(xml_file)
                continue
                
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            image_sizes.append((width, height))
            
            # 检查文件名
            filename = root.find('filename').text
            if filename is None:
                print(f"警告: {xml_file} 缺少文件名")
                error_files.append(xml_file)
                continue
            
            # 统计对象信息
            objects = list(root.iter('object'))
            if len(objects) == 0:
                print(f"警告: {xml_file} 没有标注对象")
            
            for obj in objects:
                cls_name = obj.find('name').text
                if cls_name is None:
                    print(f"警告: {xml_file} 中对象缺少类别名称")
                    continue
                
                class_distribution[cls_name] += 1
                
                # 检查边界框
                bbox = obj.find('bndbox')
                if bbox is None:
                    print(f"警告: {xml_file} 中对象缺少边界框")
                    continue
                
                try:
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # 检查边界框有效性
                    if xmin >= xmax or ymin >= ymax:
                        print(f"错误: {xml_file} 边界框坐标无效: ({xmin},{ymin},{xmax},{ymax})")
                        error_files.append(xml_file)
                        continue
                    
                    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                        print(f"警告: {xml_file} 边界框超出图像范围")
                    
                    # 计算边界框尺寸
                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    
                    bbox_stats[cls_name].append((bbox_width, bbox_height))
                    
                except (ValueError, TypeError) as e:
                    print(f"错误: {xml_file} 边界框坐标格式错误: {e}")
                    error_files.append(xml_file)
                    
        except Exception as e:
            print(f"错误: 解析 {xml_file} 时出错: {e}")
            error_files.append(xml_file)
    
    # 输出类别分布
    print("\nVOC数据集类别分布:")
    print("-" * 40)
    total_objects = sum(class_distribution.values())
    for cls_name, count in class_distribution.most_common():
        percentage = (count / total_objects) * 100
        print(f"  {cls_name}: {count} 个 ({percentage:.1f}%)")
    
    # 输出边界框统计
    print("\nVOC数据集边界框尺寸统计:")
    print("-" * 40)
    for cls_name in sorted(bbox_stats.keys()):
        widths = [w for w, h in bbox_stats[cls_name]]
        heights = [h for w, h in bbox_stats[cls_name]]
        
        if widths and heights:
            avg_width = sum(widths) / len(widths)
            avg_height = sum(heights) / len(heights)
            print(f"  {cls_name}:")
            print(f"    平均尺寸: {avg_width:.1f} x {avg_height:.1f} 像素")
            print(f"    最小尺寸: {min(widths):.1f} x {min(heights):.1f} 像素")
            print(f"    最大尺寸: {max(widths):.1f} x {max(heights):.1f} 像素")
    
    # 输出图像尺寸统计
    if image_sizes:
        print("\n图像尺寸统计:")
        print("-" * 40)
        widths = [w for w, h in image_sizes]
        heights = [h for w, h in image_sizes]
        print(f"  平均尺寸: {sum(widths)/len(widths):.1f} x {sum(heights)/len(heights):.1f}")
        print(f"  最小尺寸: {min(widths)} x {min(heights)}")
        print(f"  最大尺寸: {max(widths)} x {max(heights)}")
    
    # 检查与YOLO转换的一致性
    print("\n与YOLO转换一致性检查:")
    print("-" * 40)
    yolo_class_names = ['控制适应标识', '无螺丝', '有螺丝', '电路板']
    voc_class_names = list(class_distribution.keys())
    
    print(f"YOLO类别: {yolo_class_names}")
    print(f"VOC类别: {voc_class_names}")
    
    # 检查类别名称匹配
    missing_in_voc = set(yolo_class_names) - set(voc_class_names)
    extra_in_voc = set(voc_class_names) - set(yolo_class_names)
    
    if missing_in_voc:
        print(f"警告: 以下YOLO类别在VOC中不存在: {list(missing_in_voc)}")
    
    if extra_in_voc:
        print(f"警告: 以下VOC类别在YOLO映射中不存在: {list(extra_in_voc)}")
    
    # 总结
    print("\n检查总结:")
    print("-" * 40)
    if not missing_images and not missing_xmls and not error_files and not missing_in_voc and not extra_in_voc:
        print("✅ VOC数据集检查通过!")
        return True
    else:
        print("❌ VOC数据集存在问题:")
        if missing_images:
            print(f"  - {len(missing_images)} 个XML文件缺少对应图像")
        if missing_xmls:
            print(f"  - {len(missing_xmls)} 个图像文件缺少对应XML")
        if error_files:
            print(f"  - {len(set(error_files))} 个XML文件存在格式错误")
        if missing_in_voc:
            print(f"  - {len(missing_in_voc)} 个YOLO类别在VOC中缺失")
        if extra_in_voc:
            print(f"  - {len(extra_in_voc)} 个VOC类别在YOLO映射中缺失")
        return False

def main():
    parser = argparse.ArgumentParser(description='VOC数据集检查工具')
    parser.add_argument('--voc_annotations', required=True, help='VOC标注目录')
    parser.add_argument('--voc_images', required=True, help='VOC图像目录')
    
    args = parser.parse_args()
    
    success = check_voc_dataset(args.voc_annotations, args.voc_images)
    
    # 退出码：0表示成功，1表示失败
    exit(0 if success else 1)

if __name__ == '__main__':
    main()