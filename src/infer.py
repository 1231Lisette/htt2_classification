"""
推理模块
实现预测、PCB状态判定和可视化
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

class Inference:
    """推理器类"""
    
    def __init__(self, model_path, config):
        """
        初始化推理器
        
        Args:
            model_path: 模型权重路径
            config: 配置字典
        """
        self.model = YOLO(model_path)
        self.config = config
        self.class_names = ['控制适应标识', '无螺丝', '有螺丝', '电路板']
        
        # 颜色配置：无螺丝为红色，其他为不同颜色
        self.colors = {
            0: (0, 255, 0),    # 控制适应标识 - 绿色
            1: (0, 0, 255),    # 无螺丝 - 红色（异常）
            2: (255, 0, 0),    # 有螺丝 - 蓝色
            3: (255, 255, 0)   # 电路板 - 青色
        }
    
    def predict_single(self, image_path, output_dir=None, save_result=True):
        """
        预测单张图像
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            save_result: 是否保存结果
            
        Returns:
            result: 检测结果
            visualized_img: 可视化图像
            pcb_status: PCB状态 ('normal'/'abnormal')
        """
        # 读取图像
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 执行预测
        infer_cfg = self.config['inference']
        results = self.model.predict(
            image_path,
            conf=infer_cfg['conf_threshold'],
            iou=infer_cfg['iou_threshold'],
            imgsz=self.config['train']['img_size'],
            max_det=infer_cfg['max_det']
        )
        
        result = results[0]
        
        # 分析PCB状态
        pcb_status = self._analyze_pcb_status(result)
        
        # 可视化结果
        visualized_img = self._visualize_detections(image_path, result, pcb_status)
        
        # 保存结果
        if save_result and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'pred_{os.path.basename(image_path)}')
            cv2.imwrite(output_path, visualized_img)
            print(f"结果保存至: {output_path}")
        
        return result, visualized_img, pcb_status
    
    def predict_batch(self, image_dir, output_dir):
        """
        批量预测
        
        Args:
            image_dir: 输入图像目录
            output_dir: 输出目录
        """
        import glob
        
        image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                     glob.glob(os.path.join(image_dir, "*.png")) + \
                     glob.glob(os.path.join(image_dir, "*.jpeg"))
        
        if not image_files:
            print(f"在 {image_dir} 中未找到图像文件")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        status_count = {'normal': 0, 'abnormal': 0}
        
        for img_file in image_files:
            try:
                result, visualized_img, pcb_status = self.predict_single(
                    img_file, output_dir, save_result=True
                )
                
                status_count[pcb_status] += 1
                print(f"处理: {os.path.basename(img_file)} - 状态: {pcb_status}")
                
            except Exception as e:
                print(f"处理图像 {img_file} 时出错: {e}")
        
        print(f"\n批量处理完成!")
        print(f"正常PCB: {status_count['normal']} 个")
        print(f"异常PCB: {status_count['abnormal']} 个")
        
        return status_count
    
    def _analyze_pcb_status(self, result):
        """
        分析PCB状态
        
        Args:
            result: 检测结果
            
        Returns:
            status: 'normal' 或 'abnormal'
        """
        if not hasattr(result, 'boxes') or result.boxes is None:
            return 'unknown'
        
        boxes = result.boxes
        has_no_screw = False
        has_circuit_board = False
        
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 1:  # 无螺丝
                has_no_screw = True
            elif cls_id == 3:  # 电路板
                has_circuit_board = True
        
        # 如果有电路板且检测到无螺丝，则为异常状态
        if has_circuit_board and has_no_screw:
            return 'abnormal'
        else:
            return 'normal'
    


    def _visualize_detections(self, image_path, result, pcb_status):
        """可视化检测结果（支持中文显示）"""
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转为 PIL 图像（PIL 支持中文字体）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # 指定中文字体路径
        font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # Ubuntu/Linux
        # 如果你在 Windows，用：
        # font_path = "C:/Windows/Fonts/simhei.ttf"
        # 如果在 macOS，用：
        # font_path = "/System/Library/Fonts/STHeiti Medium.ttc"

        font = ImageFont.truetype(font_path, 20)

        # 绘制检测框
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()

                color = self.colors.get(cls_id, (255, 255, 255))
                x1, y1, x2, y2 = map(int, bbox)

                # 画矩形框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # 绘制中文标签
                label = f"{self.class_names[cls_id]} {confidence:.2f}"
                draw.text((x1 + 5, y1 - 25), label, font=font, fill=(255, 255, 255))

        # 添加 PCB 状态
        status_color = (255, 0, 0) if pcb_status == 'abnormal' else (0, 255, 0)
        status_text = f"PCB 状态: {'异常' if pcb_status == 'abnormal' else '正常'}"
        draw.text((10, 20), status_text, font=font, fill=status_color)

        # 转回 OpenCV 格式
        image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return image

    
