"""
训练模块
整合训练、验证、模型保存功能
"""

import os
import torch
import yaml
from datetime import datetime
from ultralytics import YOLO
from src.utils import setup_environment
class Trainer:
    """训练器类"""
    
    def __init__(self, config_path):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.train_results = None
    
    def _load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_training(self, custom_model_path=None):
        """
        设置训练环境
        
        Args:
            custom_model_path: 自定义模型权重路径
        """
            # 设置环境（包括中文字体）
    
        setup_environment()
        # 设置设备
        device = self.config['train'].get('device', '0')
        if device == 'cpu' or not torch.cuda.is_available():
            self.device = 'cpu'
            print("使用CPU进行训练")
        else:
            self.device = device
            print(f"使用GPU {device} 进行训练")
        
        # 创建输出目录
        save_dir = self.config['train'].get('save_dir', 'experiments/runs')
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化模型
        if custom_model_path and os.path.exists(custom_model_path):
            self.model = YOLO(custom_model_path)
            print(f"加载现有模型: {custom_model_path}")
        else:
            model_name = self.config['train'].get('model', 'yolov8m.pt')
            self.model = YOLO(model_name)
            print(f"加载预训练模型: {model_name}")
        
        # 更新模型类别数
        self.model.model.nc = self.config['model'].get('nc', 4)
        print(f"训练配置: {self.config['train']['epochs']} 轮次, {self.config['train']['batch_size']} 批次大小")
    
    def train(self):
        """执行训练"""
        if self.model is None:
            raise ValueError("模型未初始化，请先调用 setup_training()")
        
        print("开始训练...")
        
        # 训练参数
        train_cfg = self.config['train']
        
        # WandB配置
        wandb_cfg = self.config.get('wandb', {})
        if wandb_cfg.get('enabled', False):
            import wandb
            wandb.init(
                project=wandb_cfg.get('project', 'HT22_Detection'),
                name=wandb_cfg.get('name', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                tags=wandb_cfg.get('tags', []),
                config=train_cfg
            )
        
        # 执行训练
        self.train_results = self.model.train(
            data='configs/dataset.yaml',
            epochs=train_cfg['epochs'],
            imgsz=train_cfg['img_size'],
            batch=train_cfg['batch_size'],
            device=self.device,
            workers=train_cfg.get('workers', 8),
            lr0=train_cfg.get('lr0', 0.01),
            weight_decay=train_cfg.get('weight_decay', 0.0005),
            patience=train_cfg['patience'],
            save_period=train_cfg.get('save_period', 10),
            project=train_cfg['save_dir'],
            name=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            exist_ok=True
        )
        
        # 结束WandB运行
        if wandb_cfg.get('enabled', False) and wandb.run is not None:
            wandb.finish()
        
        print("训练完成!")
        return self.train_results
    
    def validate(self, data_path=None):
        """
        验证模型
        
        Args:
            data_path: 验证数据集路径
        """
        if self.model is None:
            raise ValueError("模型未初始化")
        
        data_cfg = data_path if data_path else 'configs/dataset.yaml'
        metrics = self.model.val(data=data_cfg)
        
        print(f"验证结果 - mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")
        return metrics
    
    def test(self, data_path=None):
        """
        在测试集上评估模型
        
        Args:
            data_path: 测试数据集路径
            
        Returns:
            metrics: 评估指标
        """
        if self.model is None:
            raise ValueError("模型未初始化")
        
        # 使用测试集进行评估
        data_cfg = data_path if data_path else 'configs/dataset.yaml'
        
        print("开始在测试集上评估模型...")
        
        # 注意：这里需要确保数据集配置文件中指定了测试集路径
        metrics = self.model.val(data=data_cfg, split='test')  # 使用测试集
        
        print(f"测试集结果 - mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")
        
        # 保存测试结果
        test_results = {
            'map50': float(metrics.box.map50),
            'map': float(metrics.box.map),
            'precision': float(metrics.box.p),
            'recall': float(metrics.box.r)
        }
        
        from src.utils import save_results
        save_results(test_results, 'experiments/results/test_metrics.json')
        
        return metrics
    
    def export_model(self, format='onnx', output_dir='experiments/weights'):
        """
        导出模型
        
        Args:
            format: 导出格式 ('onnx', 'torchscript', 'tflite')
            output_dir: 输出目录
        """
        if self.model is None:
            raise ValueError("模型未初始化")
        
        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, f'model.{format}')
        
        success = self.model.export(format=format, imgsz=self.config['train']['img_size'])
        if success:
            print(f"模型已导出至: {export_path}")
        else:
            print("模型导出失败")
        
        return success