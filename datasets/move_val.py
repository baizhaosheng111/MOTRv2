#!/usr/bin/env python
# 将指定目录中30%的文件夹移动到同级val目录

import os
import random
import shutil
from pathlib import Path

def move_folders(src_dir, percentage=0.3):
    """
    随机移动源目录中的文件夹到同级val目录
    
    Args:
        src_dir: 源目录路径
        percentage: 移动比例
    """
    src_path = Path(src_dir)
    # 修改目标目录路径
    dst_dir = src_path.parent / 'val'  # 直接指向val目录
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有子目录
    folders = [f for f in src_path.iterdir() if f.is_dir()]

    # 计算需要移动的数量
    num_to_move = int(len(folders) * percentage)
    if num_to_move < 1:
        print("需要移动的文件夹数量为0，请调整百分比")
        return

    # 随机选择文件夹
    selected = random.sample(folders, num_to_move)
    
    # 移动文件夹
    for folder in selected:
        dest = dst_dir / folder.name
        if not dest.exists():
            shutil.move(str(folder), str(dest))
            print(f"已移动: {folder} -> {dest}")
        else:
            print(f"跳过已存在: {dest}")

if __name__ == '__main__':
    # 修改源目录为train目录
    source_dir = '/home/xyj/MOTRv2/data/Dataset/mot/anno202503/train'
    move_folders(source_dir, 0.3)