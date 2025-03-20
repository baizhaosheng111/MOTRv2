import os
import json
import glob
from collections import defaultdict

def process_gt_file(gt_file_path):
    """处理单个gt.txt文件，按帧ID提取bbox信息"""
    frame_bboxes = defaultdict(list)
    
    with open(gt_file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 9:
            # 提取帧ID
            frame_id = int(parts[0])
            # 提取所需信息：left, top, width, height, confidence
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            confidence = float(parts[6])
            
            # 按照目标格式构建字符串
            bbox_str = f"{left},{top},{width},{height},{confidence}\n"
            frame_bboxes[frame_id].append(bbox_str)
    
    return frame_bboxes

def main():
    # 基础路径
    base_dir = "/home/xyj/MOTRv2/data/Dataset/mot/anno202503"
    output_file = "/home/xyj/MOTRv2/data/det_db_anno20250320.json"
    
    # 存储结果的字典
    result_dict = {}
    
    # 处理train、test和val文件夹
    for split in ["train", "test", "val"]:
        split_dir = os.path.join(base_dir, split)
        
        # 检查目录是否存在
        if not os.path.exists(split_dir):
            print(f"目录不存在: {split_dir}")
            continue
        
        # 查找所有序列目录
        sequence_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for seq in sequence_dirs:
            gt_file = os.path.join(split_dir, seq, "gt", "gt.txt")
            
            # 检查gt.txt文件是否存在
            if os.path.exists(gt_file):
                # 处理gt文件并获取按帧ID组织的结果
                frame_bboxes = process_gt_file(gt_file)
                
                # 将每一帧的bbox添加到结果字典中
                for frame_id, bbox_list in frame_bboxes.items():
                    # 构建图像文件路径作为键 (格式: anno202503/split/seq/img1/00000xxx.txt)
                    img_key = f"anno202503/{split}/{seq}/img1/{frame_id:08d}.txt"
                    
                    # 如果有结果，添加到字典中
                    if bbox_list:
                        result_dict[img_key] = bbox_list
    
    # 将结果写入JSON文件
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
    
    print(f"处理完成，结果已保存到: {output_file}")
    print(f"共处理了 {len(result_dict)} 个帧")

if __name__ == "__main__":
    main()