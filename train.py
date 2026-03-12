"""
使用 YOLOv8s 训练 hit 检测模型。
"""

from ultralytics import YOLO


def main():
    model = YOLO('yolov8s.pt')

    model.train(
        data='./dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project='runs/detect',
        name='hit',
        patience=20,
    )

    # 训练结束后在验证集上评估
    metrics = model.val()
    print(f'\nmAP50:   {metrics.box.map50:.4f}')
    print(f'mAP50-95: {metrics.box.map:.4f}')


if __name__ == '__main__':
    main()
