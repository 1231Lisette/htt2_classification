#!/bin/bash
echo "=== HT22数据集处理流程 ==="

# 设置变量
VOC_ANNOTATIONS="data/data_voc/Annotations"
VOC_IMAGES="data/data_voc/JPEGImages"
YOLO_ALL_DIR="data/data_yolo_all"
YOLO_SPLIT_DIR="data/data_yolo"

echo "步骤1: VOC转YOLO格式转换..."
python scripts/convert_voc_to_yolo.py \
  --voc_annotations $VOC_ANNOTATIONS \
  --voc_images $VOC_IMAGES \
  --output $YOLO_ALL_DIR

echo -e "\n步骤2: 数据完整性检查..."
python scripts/check_dataset.py --data_dir $YOLO_ALL_DIR

if [ $? -ne 0 ]; then
    echo "数据检查失败，请修复问题后重新运行"
    exit 1
fi

echo -e "\n步骤3: 数据集划分..."
python scripts/split_dataset.py \
  --input_dir $YOLO_ALL_DIR \
  --output_dir $YOLO_SPLIT_DIR \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1

echo -e "\n步骤4: 检查划分后的数据集..."
echo "=== 训练集 ==="
python scripts/check_dataset.py --data_dir $YOLO_SPLIT_DIR/train

echo -e "\n=== 验证集 ==="
python scripts/check_dataset.py --data_dir $YOLO_SPLIT_DIR/val

echo -e "\n=== 测试集 ==="
python scripts/check_dataset.py --data_dir $YOLO_SPLIT_DIR/test

echo -e "\n=== 数据集统计 ==="
echo "训练集: $(ls $YOLO_SPLIT_DIR/train/images | wc -l) 图像"
echo "验证集: $(ls $YOLO_SPLIT_DIR/val/images | wc -l) 图像"
echo "测试集: $(ls $YOLO_SPLIT_DIR/test/images | wc -l) 图像"

echo -e "\n=== 处理完成! ==="