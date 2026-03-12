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


def parse_annotations_to_dict(json_path, label_map=None):
    """解析 JSON 标注，返回 {frame_idx: [(x, y, w, h, label), ...]} 字典"""
    if label_map is None:
        label_map = {'hit': 0}

    with open(json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    frame_boxes = {}  # {frame_idx: [(x1, y1, x2, y2, label_name), ...]}

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
                label_name = labels[0]
                sequence = value.get('sequence', [])

                # 遍历关键帧序列，插值生成所有帧的框
                for i in range(len(sequence) - 1):
                    curr = sequence[i]
                    nxt = sequence[i + 1]

                    if not curr['enabled']:
                        continue

                    start_f = curr['frame']
                    end_f = nxt['frame']

                    for f_idx in range(start_f, end_f):
                        ratio = (f_idx - start_f) / (end_f - start_f) if end_f != start_f else 0

                        # Label Studio 百分比坐标
                        w_pct = _lerp(curr['width'], nxt['width'], ratio)
                        h_pct = _lerp(curr['height'], nxt['height'], ratio)
                        x_pct = _lerp(curr['x'], nxt['x'], ratio)
                        y_pct = _lerp(curr['y'], nxt['y'], ratio)

                        # 保存为百分比，绘制时再转像素
                        if f_idx not in frame_boxes:
                            frame_boxes[f_idx] = []
                        frame_boxes[f_idx].append((x_pct, y_pct, w_pct, h_pct, label_name))

    return frame_boxes


def draw_boxes_on_video(video_path, json_path, output_path):
    """在视频上绘制标注框并输出"""
    frame_boxes = parse_annotations_to_dict(json_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 如果该帧有标注，绘制框
        if frame_idx in frame_boxes:
            for (x_pct, y_pct, w_pct, h_pct, label) in frame_boxes[frame_idx]:
                # 百分比转像素坐标
                x1 = int(x_pct * width / 100)
                y1 = int(y_pct * height / 100)
                x2 = int((x_pct + w_pct) * width / 100)
                y2 = int((y_pct + h_pct) * height / 100)

                # 绘制矩形和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"视频标注完成，共 {frame_idx} 帧 -> {output_path}")


if __name__ == '__main__':
    json_path = './data/Kento_MOMOTA_CHOU_Tien_Chen_Malaysia_Open_2018_QuarterFinals/JSON/51.json'
    video_path = './data/Kento_MOMOTA_CHOU_Tien_Chen_Malaysia_Open_2018_QuarterFinals/MP4/51.mp4'
    output_video = '51_annotated.mp4'

    # 在视频上绘制标注框
    draw_boxes_on_video(video_path, json_path, output_video)
