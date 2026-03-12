"""
将 data/ 下所有视频+JSON标注 转换为 YOLOv8 检测训练格式。

输出目录结构:
  dataset/
    images/
      train/    (80%)
      val/      (20%)
    labels/
      train/
      val/
    data.yaml
"""

import json
import os
import random
import cv2

random.seed(42)
TRAIN_RATIO = 0.8

DATA_DIR = './data'
OUTPUT_DIR = './dataset'


def _lerp(a, b, t):
    return a + (b - a) * t


def get_all_video_json_pairs(data_dir):
    """扫描 data/ 目录，返回所有 (video_path, json_path, prefix) 列表"""
    pairs = []
    for match_dir in sorted(os.listdir(data_dir)):
        match_path = os.path.join(data_dir, match_dir)
        if not os.path.isdir(match_path):
            continue
        json_dir = os.path.join(match_path, 'JSON')
        mp4_dir = os.path.join(match_path, 'MP4')
        if not os.path.exists(json_dir) or not os.path.exists(mp4_dir):
            continue

        for json_name in sorted(os.listdir(json_dir)):
            if not json_name.endswith('.json') or '_filtered' in json_name:
                continue
            clip_id = json_name.replace('.json', '')
            video_path = os.path.join(mp4_dir, f'{clip_id}.mp4')
            json_path = os.path.join(json_dir, json_name)

            if not os.path.exists(video_path):
                print(f'[跳过] 视频不存在: {video_path}')
                continue

            # 用 match_dir + clip_id 作为唯一前缀，避免不同比赛的帧号冲突
            prefix = f'{match_dir}_{clip_id}'
            pairs.append((video_path, json_path, prefix))

    return pairs


def extract_frames(video_path, output_dir, prefix):
    """抽帧，返回总帧数"""
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_name = f'{prefix}_frame_{frame_idx:05d}.jpg'
        cv2.imwrite(os.path.join(output_dir, img_name), frame)
        frame_idx += 1
    cap.release()
    return frame_idx


def parse_yolo_labels(json_path, video_key=None, label_map=None):
    """
    解析 Label Studio JSON，返回 {frame_idx: [yolo_line, ...]} 字典。
    自动过滤不匹配 video_key 的 task。
    """
    if label_map is None:
        label_map = {'hit': 0}

    with open(json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    frame_labels = {}  # {frame_idx: [yolo_line, ...]}

    for task in tasks:
        # 过滤不匹配的 task
        if video_key and video_key not in task['data'].get('video', ''):
            continue

        for annotation in task.get('annotations', []):
            for result in annotation.get('result', []):
                if result['type'] != 'videorectangle':
                    continue

                value = result['value']
                labels = value.get('labels', [])
                if not labels:
                    continue
                class_id = label_map.get(labels[0], 0)
                sequence = value.get('sequence', [])

                for i in range(len(sequence) - 1):
                    curr = sequence[i]
                    nxt = sequence[i + 1]

                    if not curr['enabled']:
                        continue

                    start_f = curr['frame']
                    end_f = nxt['frame']

                    for f_idx in range(start_f, end_f):
                        ratio = (f_idx - start_f) / (end_f - start_f) if end_f != start_f else 0

                        w = _lerp(curr['width'], nxt['width'], ratio) / 100
                        h = _lerp(curr['height'], nxt['height'], ratio) / 100
                        x = _lerp(curr['x'], nxt['x'], ratio) / 100
                        y = _lerp(curr['y'], nxt['y'], ratio) / 100

                        xc = x + w / 2
                        yc = y + h / 2

                        yolo_line = f'{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}'
                        frame_labels.setdefault(f_idx, []).append(yolo_line)

    return frame_labels


def main():
    # 创建输出目录
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    pairs = get_all_video_json_pairs(DATA_DIR)
    print(f'共找到 {len(pairs)} 个视频-标注对\n')

    all_samples = []  # [(img_filename, label_lines_or_None), ...]

    for video_path, json_path, prefix in pairs:
        clip_id = prefix.rsplit('_', 1)[-1]
        print(f'处理: {prefix}')

        # 解析标注
        frame_labels = parse_yolo_labels(json_path, video_key=clip_id)
        print(f'  标注帧数: {len(frame_labels)}')

        # 抽帧
        tmp_img_dir = os.path.join(OUTPUT_DIR, 'images', '_tmp')
        os.makedirs(tmp_img_dir, exist_ok=True)
        total_frames = extract_frames(video_path, tmp_img_dir, prefix)
        print(f'  总帧数: {total_frames}')

        # 收集样本
        for f_idx in range(total_frames):
            img_name = f'{prefix}_frame_{f_idx:05d}.jpg'
            labels = frame_labels.get(f_idx, None)
            all_samples.append((img_name, labels))

        print()

    # 随机划分 train/val
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * TRAIN_RATIO)
    splits = {
        'train': all_samples[:split_idx],
        'val': all_samples[split_idx:],
    }

    tmp_img_dir = os.path.join(OUTPUT_DIR, 'images', '_tmp')
    total_labeled = 0

    for split, samples in splits.items():
        img_dir = os.path.join(OUTPUT_DIR, 'images', split)
        lbl_dir = os.path.join(OUTPUT_DIR, 'labels', split)
        labeled_count = 0

        for img_name, labels in samples:
            # 移动图片
            src = os.path.join(tmp_img_dir, img_name)
            dst = os.path.join(img_dir, img_name)
            os.rename(src, dst)

            # 写标签文件（无标注的帧不写，YOLO 视为负样本）
            if labels:
                lbl_name = img_name.replace('.jpg', '.txt')
                with open(os.path.join(lbl_dir, lbl_name), 'w') as f:
                    f.write('\n'.join(labels) + '\n')
                labeled_count += 1

        total_labeled += labeled_count
        print(f'{split}: {len(samples)} 张图片, {labeled_count} 张有标注')

    # 清理临时目录
    os.rmdir(tmp_img_dir)

    # 生成 data.yaml
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    abs_output = os.path.abspath(OUTPUT_DIR)
    with open(yaml_path, 'w') as f:
        f.write(f'path: {abs_output}\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('nc: 1\n')
        f.write("names: ['hit']\n")

    print(f'\n总样本: {len(all_samples)}, 有标注: {total_labeled}')
    print(f'数据集已保存到: {OUTPUT_DIR}/')
    print(f'配置文件: {yaml_path}')


if __name__ == '__main__':
    main()
