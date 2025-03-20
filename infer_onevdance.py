# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from copy import deepcopy
import json
import numpy as np  # 添加numpy导入

import os
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from models import build_model
from util.tool import load_model
from main import get_args_parser

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader


class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        
        # 修改这部分代码，如果检测不到键，返回None表示需要跳过这一帧
        det_key =  f_path[:-4] + '.txt'
        if det_key not in self.det_db:
            return None, None  # 返回None表示需要跳过这一帧
            
        for line in self.det_db[det_key]:
            l, t, w, h, s = list(map(float, line.split(',')))
            proposals.append([(l + w / 2) / im_w,
                              (t + h / 2) / im_h,
                              w / im_w,
                              h / im_h,
                              s])
            
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5) if proposals else torch.zeros((0, 5))

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        # 修改这部分代码，处理跳过帧的情况
        img, proposals = self.load_img_from_file(self.img_list[index])
        if img is None:  # 如果需要跳过这一帧
            # 尝试找到下一个有效的帧
            next_index = index + 1
            while next_index < len(self.img_list):
                img, proposals = self.load_img_from_file(self.img_list[next_index])
                if img is not None:
                    break
                next_index += 1
                
            # 如果所有后续帧都无效，则返回一个空的结果
            if img is None:
                # 创建一个空的图像和proposals
                empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
                return self.init_img(empty_img, torch.zeros((0, 5)))
                
        return self.init_img(img, proposals)

# 跟踪
class Detector(object):
    def __init__(self, args, model, vid):
        self.args = args
        self.detr = model

        self.vid = vid
        self.seq_num = os.path.basename(vid)
        img_list = os.listdir(os.path.join(self.args.mot_path, vid, 'img1'))
        img_list = [os.path.join(vid, 'img1', i) for i in img_list if 'jpg' in i]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.predict_path = os.path.join(self.args.output_dir, args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def detect(self, prob_threshold=0.6, area_threshold=100, vis=False):
        total_dts = 0
        total_occlusion_dts = 0

        track_instances = None
        with open(os.path.join(self.args.mot_path, self.args.det_db)) as f:
            det_db = json.load(f)
        loader = DataLoader(ListImgDataset(self.args.mot_path, self.img_list, det_db), 1, num_workers=2)
        lines = []
        for i, data in enumerate(tqdm(loader)):
            cur_img, ori_img, proposals = [d[0] for d in data]
            cur_img, proposals = cur_img.cuda(), proposals.cuda()

            # track_instances = None
            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')
            seq_h, seq_w, _ = ori_img.shape

            res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            track_instances = res['track_instances']

            dt_instances = deepcopy(track_instances)

            # filter det instances by score.
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            total_dts += len(dt_instances)

            bbox_xyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_idxes.tolist()

            save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                # 添加判断，当跟踪ID大于5时打印提示信息
                if track_id > 5:
                    print("ID大于5")
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                lines.append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))
        with open(os.path.join(self.predict_path, f'{self.seq_num}.txt'), 'w') as f:
            f.writelines(lines)
        print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=100):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        print("跟踪分数：",track_instances.scores)
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        print("obj_idxes：",track_instances.obj_idxes)
        print("新目标：",new_obj)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh) # 这一帧消失的目标
        
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device) 

        self.max_obj_id += num_new_objs # 现在最大ID数

        track_instances.disappear_time[disappeared_obj] += 1 #消失时间+1帧

        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)  # 消失状态与消失时间超过阈值

        track_instances.obj_idxes[to_del] = -1 #?


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型和权重
    detr, _, _ = build_model(args)
    detr.track_embed.score_thr = args.update_score_threshold
    detr.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    # 只测试指定的单个视频
    vid = 'train/dancetrack0001'
    det = Detector(args, model=detr, vid=vid)
    det.detect(args.score_threshold)