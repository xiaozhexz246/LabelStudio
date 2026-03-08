import json
import os
import cv2


def extract_frames(video_path, output_dir):
    """从视频中抽取所有帧为图片"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg"), frame)
        frame_idx += 1
    cap.release()
    print(f"抽帧完成，共 {frame_idx} 帧 -> {output_dir}")
    return frame_idx


def convert_label_studio_to_yolo(json_path, output_dir, label_map=None):
    """将 Label Studio 视频标注 JSON 转为 YOLOv8 检测训练标签"""
    if label_map is None:
        label_map = {'hit': 0}
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    total_labels = 0
    for task in tasks:
        annotations = task.get('annotations', [])

        for annotation in annotations:
            for result in annotation.get('result', []):
                if result['type'] != 'videorectangle':
                    continue

                value = result['value']
                labels = value.get('labels', [])
                if not labels:
                    continue
                class_id = label_map.get(labels[0], 0)
                sequence = value.get('sequence', [])

                total_labels += process_sequence(
                    sequence, class_id, output_dir
                )

    print(f"转换完成，共生成 {total_labels} 条标注 -> {output_dir}")


def process_sequence(sequence, class_id, output_dir):
    """处理单个标注的关键帧序列，线性插值生成中间帧标签"""
    count = 0
    for i in range(len(sequence) - 1):
        curr = sequence[i]
        nxt = sequence[i + 1]

        # enabled=False 表示目标从该帧起消失，跳过
        if not curr['enabled']:
            continue

        start_f = curr['frame']
        end_f = nxt['frame']

        for f_idx in range(start_f, end_f):
            ratio = (f_idx - start_f) / (end_f - start_f) if end_f != start_f else 0

            # Label Studio 坐标是百分比(0-100)，转为 YOLO 归一化(0-1)
            w = _lerp(curr['width'], nxt['width'], ratio) / 100
            h = _lerp(curr['height'], nxt['height'], ratio) / 100
            x = _lerp(curr['x'], nxt['x'], ratio) / 100
            y = _lerp(curr['y'], nxt['y'], ratio) / 100

            # 左上角 -> 中心点
            xc = x + w / 2
            yc = y + h / 2

            yolo_line = f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
            file_name = f"frame_{f_idx:05d}.txt"
            with open(os.path.join(output_dir, file_name), 'a') as f:
                f.write(yolo_line)
            count += 1
    return count


def _lerp(a, b, t):
    """线性插值"""
    return a + (b - a) * t


if __name__ == '__main__':
    json_path = '1.json'
    video_path = '64f76913-14.mp4'  # 视频文件路径，按需修改
    labels_dir = './yolo_labels'
    images_dir = './yolo_images'

    # 1. 转换标注
    convert_label_studio_to_yolo(json_path, labels_dir)

    # 2. 抽帧（需要 opencv-python: pip install opencv-python）
    # extract_frames(video_path, images_dir)
