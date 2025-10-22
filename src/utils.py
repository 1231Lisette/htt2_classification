"""
工具函数模块
整合日志、计算、文件操作和字体设置工具
"""

import os
import logging
import json
import torch
import numpy as np
from datetime import datetime

# 字体设置相关
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys

def setup_logger(name='HT22_Detector', log_dir='experiments/runs/logs'):
    """
    设置日志记录器
    
    Args:
        name: 日志器名称
        log_dir: 日志目录
        
    Returns:
        logger: 配置好的日志器
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 文件处理器
        log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        iou: IoU值
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # 计算IoU
    iou = intersection / union if union > 0 else 0.0
    
    return iou

def save_results(results, output_path):
    """
    保存评估结果
    
    Args:
        results: 结果字典
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存至: {output_path}")

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        memory_info = []
        
        for i in range(gpu_count):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            memory_info.append(f"GPU {i}: {allocated:.2f}GB / {reserved:.2f}GB")
        
        return memory_info
    else:
        return ["CUDA不可用"]

def create_experiment_dir(base_dir='experiments/runs'):
    """
    创建实验目录
    
    Args:
        base_dir: 基础目录
        
    Returns:
        exp_dir: 实验目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir

def setup_chinese_font():
    """
    设置matplotlib中文字体，解决方框显示问题
    
    Returns:
        success: 是否成功设置中文字体
    """
    try:
        # 尝试使用系统中文字体
        if sys.platform == 'linux':
            # Linux系统常见中文字体路径
            chinese_fonts = [
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # WenQuanYi Micro Hei
                '/usr/share/fonts/truetype/arphic/ukai.ttc',       # AR PL UKai
                '/usr/share/fonts/truetype/arphic/uming.ttc',      # AR PL UMing
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Droid Sans
            ]
        elif sys.platform == 'darwin':
            # macOS系统
            chinese_fonts = [
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
                '/Library/Fonts/Arial Unicode.ttf',
            ]
        else:
            # Windows系统
            chinese_fonts = [
                'C:/Windows/Fonts/simhei.ttf',  # 黑体
                'C:/Windows/Fonts/simsun.ttc',  # 宋体
                'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            ]
        
        # 查找可用的中文字体
        available_font = None
        for font_path in chinese_fonts:
            if os.path.exists(font_path):
                available_font = font_path
                break
        
        if available_font:
            # 添加字体路径到matplotlib字体管理器
            font_prop = font_manager.FontProperties(fname=available_font)
            font_name = font_prop.get_name()
            
            # 设置matplotlib字体
            matplotlib.rcParams['font.family'] = [font_name, 'sans-serif']
            matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
            
            print(f"✅ 已设置中文字体: {available_font}")
            return True
        else:
            # 如果找不到系统字体，尝试使用matplotlib的默认中文字体
            try:
                matplotlib.rcParams['font.family'] = ['sans-serif']
                matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'SimSun']
                matplotlib.rcParams['axes.unicode_minus'] = False
                print("⚠️ 使用matplotlib默认中文字体设置")
                return True
            except:
                print("⚠️ 无法设置中文字体，将使用默认字体")
                return False
            
    except Exception as e:
        print(f"❌ 设置中文字体失败: {e}")
        return False

def test_chinese_font():
    """
    测试中文字体是否正常工作
    
    Returns:
        success: 测试是否成功
    """
    try:
        # 先设置字体
        font_success = setup_chinese_font()
        if not font_success:
            return False
            
        # 创建测试图像
        plt.figure(figsize=(10, 6))
        
        # 测试所有类别名称的显示
        class_names = ['控制适应标识', '无螺丝', '有螺丝', '电路板']
        x = range(len(class_names))
        y = [1, 2, 3, 4]
        
        plt.bar(x, y)
        plt.title('中文字体测试 - HT22控制框检测类别')
        plt.xlabel('类别名称')
        plt.ylabel('测试数值')
        plt.xticks(x, class_names, rotation=45)
        
        # 添加图例
        for i, name in enumerate(class_names):
            plt.text(i, y[i] + 0.1, name, ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存测试图像
        test_dir = 'experiments/font_test'
        os.makedirs(test_dir, exist_ok=True)
        test_path = os.path.join(test_dir, 'chinese_font_test.png')
        plt.savefig(test_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 中文字体测试完成，查看: {test_path}")
        return True
        
    except Exception as e:
        print(f"❌ 中文字体测试失败: {e}")
        return False

def setup_environment():
    """
    设置完整的环境（字体、日志等）
    
    Returns:
        success: 环境设置是否成功
    """
    print("=== 设置环境 ===")
    
    # 设置日志
    logger = setup_logger()
    logger.info("环境设置开始")
    
    # 设置中文字体
    font_success = setup_chinese_font()
    if font_success:
        logger.info("中文字体设置成功")
    else:
        logger.warning("中文字体设置失败，图表中可能出现方框")
    
    # 检查GPU
    gpu_info = check_gpu_memory()
    for info in gpu_info:
        logger.info(info)
    
    # 创建实验目录
    exp_dir = create_experiment_dir()
    logger.info(f"实验目录: {exp_dir}")
    
    return font_success