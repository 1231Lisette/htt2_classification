#!/usr/bin/env python3
"""
HT22控制框检测与分类 - 主程序入口
支持训练、推理、评估等多种模式
"""

import argparse
import os
import sys

# 添加src到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import Trainer
from src.infer import Inference
from src.utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description='HT22控制框检测与分类')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'infer', 'eval', 'export', 'test'],  # 添加 'test'
                       help='运行模式: train, infer, eval, export, test')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--weights', type=str, 
                       help='模型权重路径 (推理和评估模式需要)')
    parser.add_argument('--source', type=str, 
                       help='输入源 (图像文件或目录路径)')
    parser.add_argument('--output', type=str, default='experiments/results',
                       help='输出目录路径')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger()
    logger.info(f"启动模式: {args.mode}")
    
    try:
        if args.mode == 'train':
            # 训练模式
            logger.info("进入训练模式")
            trainer = Trainer(args.config)
            trainer.setup_training()
            results = trainer.train()
            
            # 验证训练结果（在验证集上）
            logger.info("开始模型验证")
            metrics = trainer.validate()
            
            logger.info("训练流程完成")
            
        elif args.mode == 'test':
            # 测试模式 - 在测试集上评估
            if not args.weights:
                raise ValueError("测试模式需要指定模型权重路径 --weights")
            
            logger.info("进入测试模式")
            trainer = Trainer(args.config)
            trainer.setup_training(custom_model_path=args.weights)
            
            # 在测试集上评估
            test_metrics = trainer.test()  # 需要确保trainer有test方法
            
            logger.info(f"测试完成 - mAP50: {test_metrics.box.map50:.4f}")
            
        elif args.mode == 'infer':
            # 推理模式
            if not args.weights:
                raise ValueError("推理模式需要指定模型权重路径 --weights")
            if not args.source:
                raise ValueError("推理模式需要指定输入源 --source")
            
            logger.info("进入推理模式")
            import yaml
            
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            inferencer = Inference(args.weights, config)
            
            # 判断输入类型 (文件或目录)
            if os.path.isfile(args.source):
                logger.info(f"处理单张图像: {args.source}")
                result, visualized_img, status = inferencer.predict_single(
                    args.source, args.output
                )
                logger.info(f"检测完成 - PCB状态: {status}")
            else:
                logger.info(f"处理图像目录: {args.source}")
                status_count = inferencer.predict_batch(args.source, args.output)
                logger.info(f"批量处理完成 - 正常: {status_count['normal']}, 异常: {status_count['abnormal']}")
            
        elif args.mode == 'eval':
            # 评估模式（在验证集上）
            if not args.weights:
                raise ValueError("评估模式需要指定模型权重路径 --weights")
            
            logger.info("进入评估模式")
            trainer = Trainer(args.config)
            trainer.setup_training(custom_model_path=args.weights)
            metrics = trainer.validate()  # 在验证集上评估
            
            logger.info(f"评估完成 - mAP50: {metrics.box.map50:.4f}")
            
        elif args.mode == 'export':
            # 模型导出模式
            if not args.weights:
                raise ValueError("导出模式需要指定模型权重路径 --weights")
            
            logger.info("进入模型导出模式")
            trainer = Trainer(args.config)
            trainer.setup_training(custom_model_path=args.weights)
            success = trainer.export_model(output_dir=args.output)
            
            if success:
                logger.info("模型导出成功")
            else:
                logger.error("模型导出失败")
                
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == '__main__':
    main()