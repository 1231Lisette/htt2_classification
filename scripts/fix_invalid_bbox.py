#!/usr/bin/env python3
"""
高级修复无效边界框XML文件
处理各种边界框问题
"""

import os
import xml.etree.ElementTree as ET
import argparse

def fix_invalid_bbox_advanced(xml_path):
    """高级修复无效边界框"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        print(f"修复文件: {os.path.basename(xml_path)}")
        
        # 获取图像尺寸
        size = root.find('size')
        if size is None:
            print("  错误: 缺少图像尺寸信息")
            return False
            
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        print(f"  图像尺寸: {img_width}x{img_height}")
        
        objects_to_remove = []
        fixed_count = 0
        
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
                
            xmin = bbox.find('xmin')
            ymin = bbox.find('ymin')
            xmax = bbox.find('xmax')
            ymax = bbox.find('ymax')
            
            if all(elem is not None for elem in [xmin, ymin, xmax, ymax]):
                try:
                    xmin_val = float(xmin.text)
                    ymin_val = float(ymin.text)
                    xmax_val = float(xmax.text)
                    ymax_val = float(ymax.text)
                    
                    # 检查各种无效情况
                    is_invalid = False
                    
                    # 情况1: xmin >= xmax
                    if xmin_val >= xmax_val:
                        print(f"  发现无效边界框: xmin({xmin_val}) >= xmax({xmax_val})")
                        is_invalid = True
                    
                    # 情况2: ymin >= ymax
                    elif ymin_val >= ymax_val:
                        print(f"  发现无效边界框: ymin({ymin_val}) >= ymax({ymax_val})")
                        is_invalid = True
                    
                    # 情况3: 边界框超出图像范围
                    elif (xmin_val < 0 or ymin_val < 0 or 
                          xmax_val > img_width or ymax_val > img_height):
                        print(f"  发现超出图像范围的边界框: ({xmin_val}, {ymin_val}, {xmax_val}, {ymax_val})")
                        is_invalid = True
                    
                    # 情况4: 边界框面积太小（小于4像素）
                    elif (xmax_val - xmin_val) * (ymax_val - ymin_val) < 4:
                        print(f"  发现过小边界框: 面积小于4像素")
                        is_invalid = True
                    
                    if is_invalid:
                        # 标记为需要删除
                        objects_to_remove.append(obj)
                        fixed_count += 1
                        
                except ValueError as e:
                    print(f"  数值转换错误: {e}")
                    objects_to_remove.append(obj)
                    fixed_count += 1
        
        # 删除无效对象
        for obj in objects_to_remove:
            root.remove(obj)
            print(f"  已删除无效对象: {obj.find('name').text if obj.find('name') is not None else '未知类别'}")
        
        if fixed_count > 0:
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            print(f"  成功修复 {fixed_count} 个无效边界框")
        else:
            print("  无需修复")
            
        return True
        
    except Exception as e:
        print(f"修复文件时出错: {e}")
        return False

def check_and_fix_all_voc_files(voc_annotations_dir):
    """检查并修复所有VOC文件"""
    print("检查并修复所有VOC标注文件...")
    
    total_fixed = 0
    for xml_file in os.listdir(voc_annotations_dir):
        if not xml_file.endswith('.xml'):
            continue
            
        xml_path = os.path.join(voc_annotations_dir, xml_file)
        if fix_invalid_bbox_advanced(xml_path):
            total_fixed += 1
    
    print(f"\n总计修复了 {total_fixed} 个文件")

def main():
    parser = argparse.ArgumentParser(description='高级修复VOC无效边界框')
    parser.add_argument('--voc_annotations', help='VOC标注目录（修复所有文件）')
    parser.add_argument('--specific_file', help='特定文件路径（修复单个文件）')
    
    args = parser.parse_args()
    
    if args.specific_file:
        if os.path.exists(args.specific_file):
            fix_invalid_bbox_advanced(args.specific_file)
        else:
            print(f"文件不存在: {args.specific_file}")
    elif args.voc_annotations:
        if os.path.exists(args.voc_annotations):
            check_and_fix_all_voc_files(args.voc_annotations)
        else:
            print(f"目录不存在: {args.voc_annotations}")
    else:
        # 修复特定的问题文件
        xml_path = "data/data_voc/Annotations/cc_36e62fdb-frame_01494.xml"
        if os.path.exists(xml_path):
            fix_invalid_bbox_advanced(xml_path)
        else:
            print(f"默认文件不存在: {xml_path}")

if __name__ == '__main__':
    main()