# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

from collections import defaultdict
from glob import glob
import json
import os
import cv2
from tqdm import tqdm
import argparse

def get_color(i):
    return [(i * 23 * j + 43) % 255 for j in range(3)]

def process(trk_path, img_dir, output_dir, video_name):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像列表
    img_list = sorted(glob(f"{img_dir}/*.jpg"))
    if not img_list:
        print(f"未找到图像文件在 {img_dir}")
        return
    
    # 读取跟踪结果
    tracklets = defaultdict(list)
    for line in open(trk_path):
        parts = line.strip().split(',')
        if len(parts) < 7:
            continue
        t, id, x, y, w, h = map(float, parts[:6])
        t, id = int(t), int(id)
        tracklets[t].append((id, int(x), int(y), int(x+w), int(y+h)))
    
    # 处理每一帧图像
    for i, path in enumerate(tqdm(img_list, desc=f"处理 {video_name} 的图像")):
        frame_idx = i + 1  # 帧索引从1开始
        im = cv2.imread(path)
        if im is None:
            print(f"无法读取图像: {path}")
            continue
        
        # 绘制当前帧的所有跟踪框
        if frame_idx in tracklets:
            for j, x1, y1, x2, y2 in tracklets[frame_idx]:
                color = (0, 0, 255)  # BGR格式红色
                im = cv2.rectangle(im, (x1, y1), (x2, y2), color, 4)
                # 调整文字位置到框上方（y1 - 10）
                im = cv2.putText(im, f"ID:{j}", (x1 + 5, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        # 保存图像
        output_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(output_path, im)
    
    print(f"已完成 {video_name} 的可视化，结果保存在 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='可视化MOT跟踪结果')
    parser.add_argument('--video_name', type=str, default='dancetrack0001',
                        help='视频名称')
    args = parser.parse_args()
    
    video_name = args.video_name
    trk_path = f"/home/xyj/MOTRv2/tracker/{video_name}.txt"
    img_dir = f"/home/xyj/MOTRv2/data/Dataset/mot/DanceTrack/train/{video_name}/img1"
    output_dir = f"/home/xyj/MOTRv2/visualize/{video_name}/img1"
    
    # 检查文件和目录是否存在
    if not os.path.exists(trk_path):
        print(f"跟踪文件不存在: {trk_path}")
        return
    
    if not os.path.exists(img_dir):
        print(f"图像目录不存在: {img_dir}")
        return
    
    process(trk_path, img_dir, output_dir, video_name)

if __name__ == '__main__':
    main()