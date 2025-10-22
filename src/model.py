"""
模型封装模块
封装YOLOv8模型，提供统一接口
"""

from ultralytics import YOLO
import torch

class Detector:
    """检测器封装类"""
    
    def __init__(self, model_cfg, device='cuda'):
        """
        初始化检测器
        
        Args:
            model_cfg: 模型配置字典
            device: 运行设备
        """
        self.model_cfg = model_cfg
        self.device = device
        self.model = None
    
    def build_model(self, pretrained=True, weights_path=None):
        """
        构建模型
        
        Args:
            pretrained: 是否使用预训练权重
            weights_path: 自定义权重路径
        """
        if weights_path:
            # 加载自定义权重
            self.model = YOLO(weights_path)
            print(f"加载自定义权重: {weights_path}")
        elif pretrained:
            # 加载预训练模型
            model_name = self.model_cfg.get('model', 'yolov8m.pt')
            self.model = YOLO(model_name)
            print(f"加载预训练模型: {model_name}")
        else:
            # 从零开始训练（不推荐）
            self.model = YOLO('yolov8n.yaml')  # 使用配置文件
            print("从零开始初始化模型")
        
        # 更新模型类别数
        self.model.model.nc = self.model_cfg.get('nc', 4)
        print(f"模型类别数设置为: {self.model.model.nc}")
        
        return self.model
    
    def to_device(self, device):
        """移动模型到指定设备"""
        self.device = device
        if self.model:
            self.model.model.to(device)
    
    def get_model_info(self):
        """获取模型信息"""
        if not self.model:
            return "模型未初始化"
        
        info = {
            'parameters': sum(p.numel() for p in self.model.model.parameters()),
            'class_count': self.model.model.nc,
            'device': str(next(self.model.model.parameters()).device)
        }
        return info