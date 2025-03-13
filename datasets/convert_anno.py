#!/usr/bin/env python
# 将anno202503数据集转换为DanceTrack格式并生成检测数据库

import os
import json
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='/home/xyj/MOTRv2/data/Dataset/mot/trackanno2503/anno202503')
    parser.add_argument('--dst_path', type=str, default='/home/xyj/MOTRv2/data/Dataset/mot/anno202503')
    return parser.parse_args()

def convert_sequence(seq_path, output_path):
    """转换单个序列的数据"""
    # 创建目录结构
    img_dir = output_path / 'img1'
    gt_dir = output_path / 'gt'
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查可能的图像目录
    possible_img_dirs = [
        seq_path / 'img',
        seq_path / 'images',
        seq_path / 'image',
        seq_path,  # 图像可能直接在序列目录下
        seq_path / 'frames'
    ]
    
    img_src_dir = None
    for dir_path in possible_img_dirs:
        if dir_path.exists() and (any(dir_path.glob('*.jpg')) or any(dir_path.glob('*.png'))):
            img_src_dir = dir_path
            break
    
    if img_src_dir is None:
        print(f"警告: 在 {seq_path} 中找不到图像文件")
        return False
    
    # 收集所有txt文件和json文件
    txt_dir = seq_path / 'txt'
    json_dir = seq_path / 'json'
    
    if not txt_dir.exists() and not json_dir.exists():
        print(f"警告: 在 {seq_path} 中找不到txt或json标注文件")
        return False
    
    # 复制图像文件
    img_files = list(img_src_dir.glob('*.jpg')) + list(img_src_dir.glob('*.png'))
    if not img_files:
        print(f"警告: 在 {img_src_dir} 中找不到图像文件")
        return False
    
    # 创建帧ID映射
    frame_map = {}  # 原始文件名到帧ID的映射
    
    for i, img_file in enumerate(sorted(img_files, key=lambda x: int(x.stem) if x.stem.isdigit() else 0)):
        try:
            frame_id = int(img_file.stem)
            dst_img_path = img_dir / f'{frame_id:08d}.jpg'
            frame_map[img_file.stem] = frame_id
            
            # 如果是PNG，转换为JPG
            if img_file.suffix.lower() == '.png':
                img = cv2.imread(str(img_file))
                cv2.imwrite(str(dst_img_path), img)
            else:
                shutil.copy(img_file, dst_img_path)
        except ValueError:
            # 如果文件名不是数字，使用索引作为帧号
            frame_id = i + 1
            dst_img_path = img_dir / f'{frame_id:08d}.jpg'
            frame_map[img_file.stem] = frame_id
            
            if img_file.suffix.lower() == '.png':
                img = cv2.imread(str(img_file))
                cv2.imwrite(str(dst_img_path), img)
            else:
                shutil.copy(img_file, dst_img_path)
            
            print(f"警告: 图像文件名 {img_file.name} 不是数字，使用索引 {frame_id} 作为帧号")
    
    # 转换标注文件
    gt_lines_dict = {}  # 使用字典存储每个帧的标注，键为帧ID
    
    # 优先处理json文件
    if json_dir.exists():
        for json_file in json_dir.glob('*.json'):
            try:
                frame_name = json_file.stem
                if frame_name not in frame_map:
                    print(f"警告: 找不到与 {json_file} 对应的图像")
                    continue
                
                frame_id = frame_map[frame_name]
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    # 处理json格式的标注
                    if 'shapes' in data:
                        for shape in data['shapes']:
                            if shape['shape_type'] == 'rectangle':
                                points = shape['points']
                                if len(points) == 2:  # 如果只有两个点(左上和右下)
                                    x1, y1 = points[0]
                                    x2, y2 = points[1]
                                    x = min(x1, x2)
                                    y = min(y1, y2)
                                    width = abs(x2 - x1)
                                    height = abs(y2 - y1)
                                elif len(points) == 4:  # 如果有四个点(四个角)
                                    x_coords = [p[0] for p in points]
                                    y_coords = [p[1] for p in points]
                                    x = min(x_coords)
                                    y = min(y_coords)
                                    width = max(x_coords) - x
                                    height = max(y_coords) - y
                                else:
                                    continue
                                
                                # 获取跟踪ID和类别
                                track_id = shape.get('group_id', 0)
                                class_id = int(shape.get('label', 0))
                                confidence = 1.0  # 默认置信度
                                
                                # 添加到gt_lines_dict字典
                                if frame_id not in gt_lines_dict:
                                    gt_lines_dict[frame_id] = []
                                
                                gt_line = f"{frame_id},{track_id},{x},{y},{width},{height},1,{class_id},1\n"
                                gt_lines_dict[frame_id].append(gt_line)
            except Exception as e:
                print(f"处理json文件 {json_file} 时出错: {e}")
                continue
    
    # 处理txt文件，只处理json文件中没有的帧
    if txt_dir.exists():
        for txt_file in txt_dir.glob('*.txt'):
            try:
                frame_name = txt_file.stem
                if frame_name not in frame_map:
                    print(f"警告: 找不到与 {txt_file} 对应的图像")
                    continue
                
                frame_id = frame_map[frame_name]
                
                # 如果json文件已经处理了这一帧，则跳过
                if frame_id in gt_lines_dict and gt_lines_dict[frame_id]:
                    continue
                
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 15:
                            continue
                        
                        # 使用中心点和宽高计算左上角坐标
                        center_x = float(parts[10])
                        center_y = float(parts[11])
                        width = float(parts[12])
                        height = float(parts[13])
                        
                        # 计算左上角坐标
                        x = center_x - width / 2
                        y = center_y - height / 2
                        
                        # 在标注处理部分添加类型转换
                        track_id = int(float(parts[1]))  # 原第104行附近
                        frame_id = int(float(parts[0]))   # MOT格式帧号转换
                        class_id = int(parts[2])  # 类别ID
                        confidence = float(parts[14])
                        
                        # 添加到gt_lines_dict字典
                        if frame_id not in gt_lines_dict:
                            gt_lines_dict[frame_id] = []
                        
                        gt_line = f"{frame_id},{track_id},{x},{y},{width},{height},{confidence},{class_id},1\n"
                        gt_lines_dict[frame_id].append(gt_line)
            except ValueError as e:
                print(f"警告: 处理txt文件 {txt_file.name} 时出错: {e}")
                continue
    
    # 将所有标注合并到一个列表
    gt_lines = []
    for frame_id in sorted(gt_lines_dict.keys()):
        gt_lines.extend(gt_lines_dict[frame_id])
    
    if not gt_lines:
        print(f"警告: 无法从 {seq_path} 提取标注信息")
        return False
    
    # 写入gt.txt文件
    with open(gt_dir / 'gt.txt', 'w') as f:
        f.writelines(gt_lines)
    
    # 创建seqinfo.ini文件
    img_files = list(img_dir.glob('*.jpg'))
    if img_files:
        sample_img = img_files[0]
        # 获取图像尺寸
        img = cv2.imread(str(sample_img))
        if img is None:
            print(f"警告: 无法读取图像 {sample_img}")
            return False
            
        height, width = img.shape[:2]
        
        seq_name = seq_path.name
        seq_length = len(img_files)
        
        seqinfo_content = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate=30
seqLength={seq_length}
imWidth={width}
imHeight={height}
imExt=.jpg
"""
        with open(output_path / 'seqinfo.ini', 'w') as f:
            f.write(seqinfo_content)
    
    return True

def generate_det_db(dst_path, output_json):
    """生成检测数据库"""
    det_db = {}
    
    # 遍历所有序列
    for split in ['train', 'val', 'test']:
        split_path = dst_path / split
        if not split_path.exists():
            continue
        
        for seq_dir in split_path.iterdir():
            if not seq_dir.is_dir() or seq_dir.name == 'seqmap':
                continue
            
            # 读取gt.txt文件
            gt_file = seq_dir / 'gt' / 'gt.txt'
            if not gt_file.exists():
                continue
            
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 9:
                        continue
                    
                    frame_id = int(parts[0])
                    x, y, w, h = map(float, parts[2:6])
                    confidence = float(parts[6])
                    
                    # 构建键名
                    key = f"{split}/{seq_dir.name}/img1/{frame_id:08d}.txt"
                    
                    if key not in det_db:
                        det_db[key] = []
                    
                    # 添加检测结果
                    det_db[key].append(f"{x},{y},{w},{h},{confidence}")
    
    # 保存检测数据库
    with open(output_json, 'w') as f:
        json.dump(det_db, f)
    
    print(f"检测数据库已保存至: {output_json}")
    print(f"共包含 {len(det_db)} 个条目")

def main():
    args = parse_args()
    src_path = Path(args.src_path)
    dst_path = Path(args.dst_path)
    
    # 创建目标目录结构
    for split in ['train', 'val', 'test']:
        (dst_path / split).mkdir(parents=True, exist_ok=True)
    
    # 默认将所有序列放入训练集
    train_path = dst_path / 'train'
    
    # 遍历所有序列
    sequences = [d for d in src_path.iterdir() if d.is_dir()]
    success_count = 0
    
    for seq_path in tqdm(sequences, desc="处理序列"):
        seq_name = seq_path.name
        output_path = train_path / seq_name
        
        print(f"处理序列: {seq_name}")
        success = convert_sequence(seq_path, output_path)
        if success:
            success_count += 1
        else:
            print(f"转换序列 {seq_name} 失败")
    
    print(f"成功转换 {success_count}/{len(sequences)} 个序列")
    
    # 生成检测数据库
    det_db_path = '/home/xyj/MOTRv2/data/det_db_anno202503.json'
    generate_det_db(dst_path, det_db_path)

if __name__ == '__main__':
    main()