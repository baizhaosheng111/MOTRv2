import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# 定义路径
SOURCE_DIR = '/home/xyj/MOTRv2/data/Dataset/mot/weimi250322'
TARGET_DIR = '/home/xyj/MOTRv2/data/Dataset/mot/weimi250322train'

# 创建目标目录结构
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

def convert_json_to_gt(json_path):
    """将JSON文件转换为gt.txt格式的一行"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 获取帧号（从文件名中提取）
    frame_id = int(os.path.basename(json_path).split('.')[0])
    
    results = []
    for shape in data.get('shapes', []):
        # 获取跟踪ID
        track_id = shape.get('group_id', 0)
        
        # 获取矩形框的四个点坐标
        points = shape.get('points', [])
        if len(points) != 4:
            continue
        
        # 计算左上角坐标和宽高
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        left = min(x_coords)
        top = min(y_coords)
        width = max(x_coords) - left
        height = max(y_coords) - top
        
        # 格式：帧序号，跟踪id，bbox左上角left坐标，bbox左上角top坐标，bbox的宽度，bbox的高度，1，1，1
        gt_line = f"{frame_id},{track_id},{left:.1f},{top:.1f},{width:.1f},{height:.1f},1,1,1"
        results.append(gt_line)
    
    return results

def process_sequence(seq_dir, target_seq_name, split):
    """处理一个序列的数据"""
    # 创建目标目录
    target_seq_dir = os.path.join(TARGET_DIR, split, target_seq_name)
    os.makedirs(target_seq_dir, exist_ok=True)
    os.makedirs(os.path.join(target_seq_dir, 'img1'), exist_ok=True)
    os.makedirs(os.path.join(target_seq_dir, 'gt'), exist_ok=True)
    
    # 处理JSON文件，生成gt.txt
    json_dir = os.path.join(seq_dir, 'json')
    gt_lines = []
    
    if os.path.exists(json_dir):
        for json_file in tqdm(os.listdir(json_dir), desc=f"处理 {target_seq_name} 的JSON文件"):
            if json_file.endswith('.json'):
                json_path = os.path.join(json_dir, json_file)
                gt_lines.extend(convert_json_to_gt(json_path))
    
    # 写入gt.txt
    with open(os.path.join(target_seq_dir, 'gt', 'gt.txt'), 'w') as f:
        for line in sorted(gt_lines, key=lambda x: (int(x.split(',')[0]), int(x.split(',')[1]))):
            f.write(line + '\n')
    
    # 复制图片文件
    img_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        img_files.extend(list(Path(seq_dir).glob(f'*{ext}')))
    
    for img_file in tqdm(img_files, desc=f"复制 {target_seq_name} 的图片文件"):
        shutil.copy(img_file, os.path.join(target_seq_dir, 'img1', img_file.name))

def main():
    # 获取所有序列目录
    seq_dirs = []
    for item in os.listdir(SOURCE_DIR):
        item_path = os.path.join(SOURCE_DIR, item)
        if os.path.isdir(item_path):
            seq_dirs.append(item_path)
    
    # 随机打乱序列顺序
    random.shuffle(seq_dirs)
    
    # 按6:3:1的比例分配
    total = len(seq_dirs)
    train_count = int(total * 0.6)
    val_count = int(total * 0.3)
    
    train_seqs = seq_dirs[:train_count]
    val_seqs = seq_dirs[train_count:train_count+val_count]
    test_seqs = seq_dirs[train_count+val_count:]
    
    # 处理训练集
    for i, seq_dir in enumerate(train_seqs):
        seq_name = f"weimi_train_{i+1:04d}"
        process_sequence(seq_dir, seq_name, 'train')
    
    # 处理验证集
    for i, seq_dir in enumerate(val_seqs):
        seq_name = f"weimi_val_{i+1:04d}"
        process_sequence(seq_dir, seq_name, 'val')
    
    # 处理测试集
    for i, seq_dir in enumerate(test_seqs):
        seq_name = f"weimi_test_{i+1:04d}"
        process_sequence(seq_dir, seq_name, 'test')
    
    print(f"数据转换完成！")
    print(f"训练集: {len(train_seqs)} 个序列")
    print(f"验证集: {len(val_seqs)} 个序列")
    print(f"测试集: {len(test_seqs)} 个序列")

if __name__ == "__main__":
    main()