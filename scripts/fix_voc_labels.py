#!/usr/bin/env python3
"""
修正VOC数据集中错误的类别标签
将"无螺丝"和"电路板"的标签互换
"""

import os
import xml.etree.ElementTree as ET
import argparse
import shutil

def fix_voc_labels(voc_annotations_dir, output_dir=None):
    """
    修正VOC标注文件中的类别标签错误
    
    Args:
        voc_annotations_dir: VOC标注文件目录
        output_dir: 输出目录（如果为None，则原地修改）
    """
    if output_dir is None:
        output_dir = voc_annotations_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print("开始修正VOC标签...")
    print("修正规则: '无螺丝' ↔ '电路板'")
    
    processed_count = 0
    fixed_count = 0
    error_count = 0
    
    for xml_file in os.listdir(voc_annotations_dir):
        if not xml_file.endswith('.xml'):
            continue
            
        xml_path = os.path.join(voc_annotations_dir, xml_file)
        output_path = os.path.join(output_dir, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            modified = False
            
            # 遍历所有对象
            for obj in root.iter('object'):
                cls_name = obj.find('name')
                if cls_name is None:
                    continue
                
                # 修正类别标签
                if cls_name.text == '无螺丝':
                    cls_name.text = '电路板'
                    modified = True
                    fixed_count += 1
                elif cls_name.text == '电路板':
                    cls_name.text = '无螺丝'
                    modified = True
                    fixed_count += 1
            
            # 保存修正后的XML文件
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            processed_count += 1
            
            # 修复无效边界框（针对cc_36e62fdb-frame_01494.xml）
            if xml_file == 'cc_36e62fdb-frame_01494.xml':
                fix_invalid_bbox(output_path)
                print(f"已修复无效边界框: {xml_file}")
            
        except Exception as e:
            print(f"处理文件 {xml_file} 时出错: {e}")
            error_count += 1#!/usr/bin/env python3
"""
修正VOC数据集中错误的类别标签
将"无螺丝"和"电路板"的标签互换
"""

import os
import xml.etree.ElementTree as ET
import argparse
import shutil

def fix_voc_labels(voc_annotations_dir, output_dir=None):
    """
    修正VOC标注文件中的类别标签错误
    
    Args:
        voc_annotations_dir: VOC标注文件目录
        output_dir: 输出目录（如果为None，则原地修改）
    """
    if output_dir is None:
        output_dir = voc_annotations_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print("开始修正VOC标签...")
    print("修正规则: '无螺丝' ↔ '电路板'")
    
    processed_count = 0
    fixed_count = 0
    error_count = 0
    
    for xml_file in os.listdir(voc_annotations_dir):
        if not xml_file.endswith('.xml'):
            continue
            
        xml_path = os.path.join(voc_annotations_dir, xml_file)
        output_path = os.path.join(output_dir, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            modified = False
            
            # 遍历所有对象
            for obj in root.iter('object'):
                cls_name = obj.find('name')
                if cls_name is None:
                    continue
                
                # 修正类别标签
                if cls_name.text == '无螺丝':
                    cls_name.text = '电路板'
                    modified = True
                    fixed_count += 1
                elif cls_name.text == '电路板':
                    cls_name.text = '无螺丝'
                    modified = True
                    fixed_count += 1
            
            # 保存修正后的XML文件
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            processed_count += 1
            
            # 修复无效边界框（针对cc_36e62fdb-frame_01494.xml）
            if xml_file == 'cc_36e62fdb-frame_01494.xml':
                fix_invalid_bbox(output_path)
                print(f"已修复无效边界框: {xml_file}")
            
        except Exception as e:
            print(f"处理文件 {xml_file} 时出错: {e}")
            error_count += 1
    
    print(f"\n修正完成!")
    print(f"处理文件: {processed_count} 个")
    print(f"修正标签: {fixed_count} 个")
    print(f"错误文件: {error_count} 个")
    print(f"输出目录: {output_dir}")

def fix_invalid_bbox(xml_path):
    """修复特定的无效边界框"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
                
            xmin = bbox.find('xmin')
            ymin = bbox.find('ymin')
            xmax = bbox.find('xmax')
            ymax = bbox.find('ymax')
            
            # 检查是否xmin >= xmax
            if (xmin is not None and xmax is not None and 
                float(xmin.text) >= float(xmax.text)):
                # 交换xmin和xmax
                xmin_text = xmin.text
                xmin.text = xmax.text
                xmax.text = xmin_text
                print(f"  修复边界框: ({xmin.text}, {ymin.text}, {xmax.text}, {ymax.text})")
        
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
    except Exception as e:
        print(f"修复边界框时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='修正VOC标签错误')
    parser.add_argument('--voc_annotations', required=True, help='VOC标注文件目录')
    parser.add_argument('--output_dir', help='输出目录（如果提供，则保存到新目录；否则原地修改）')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.voc_annotations):
        print(f"错误: VOC标注目录不存在: {args.voc_annotations}")
        return
    
    fix_voc_labels(args.voc_annotations, args.output_dir)

if __name__ == '__main__':
    main()
    
    print(f"\n修正完成!")
    print(f"处理文件: {processed_count} 个")
    print(f"修正标签: {fixed_count} 个")
    print(f"错误文件: {error_count} 个")
    print(f"输出目录: {output_dir}")

def fix_invalid_bbox(xml_path):
    """修复特定的无效边界框"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
                
            xmin = bbox.find('xmin')
            ymin = bbox.find('ymin')
            xmax = bbox.find('xmax')
            ymax = bbox.find('ymax')
            
            # 检查是否xmin >= xmax
            if (xmin is not None and xmax is not None and 
                float(xmin.text) >= float(xmax.text)):
                # 交换xmin和xmax
                xmin_text = xmin.text
                xmin.text = xmax.text
                xmax.text = xmin_text
                print(f"  修复边界框: ({xmin.text}, {ymin.text}, {xmax.text}, {ymax.text})")
        
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
    except Exception as e:
        print(f"修复边界框时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='修正VOC标签错误')
    parser.add_argument('--voc_annotations', required=True, help='VOC标注文件目录')
    parser.add_argument('--output_dir', help='输出目录（如果提供，则保存到新目录；否则原地修改）')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.voc_annotations):
        print(f"错误: VOC标注目录不存在: {args.voc_annotations}")
        return
    
    fix_voc_labels(args.voc_annotations, args.output_dir)

if __name__ == '__main__':
    main()