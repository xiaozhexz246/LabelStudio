import json

base = './data/CHOU_Tien_Chen_Jonatan_CHRISTIE_Sudirman_Cup_2019_Quarter-finals/JSON'

filters = [
    ('69.json', '69'),
    ('356.json', '356'),
]

for filename, video_key in filters:
    input_path = f'{base}/{filename}'
    output_path = f'{base}/{filename.replace(".json", "_filtered.json")}'

    with open(input_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    filtered = [t for t in tasks if video_key in t['data']['video']]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f'{filename}: {len(tasks)} -> {len(filtered)} task, 保存到 {output_path}')
