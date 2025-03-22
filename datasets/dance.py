# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
MOT dataset which returns image_id for evaluation.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances

from random import choice, randint


def is_crowd(ann):
    return 'extra' in ann and 'ignore' in ann['extra'] and ann['extra']['ignore'] == 1


class DetMOTDetection:
    # 修改__init__方法，移除CrowdHuman相关代码
    def __init__(self, args, data_txt_path: str, seqs_folder, transform):
        self.args = args
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.mot_path = args.mot_path
    
        self.labels_full = defaultdict(lambda : defaultdict(list))
        def add_mot_folder(split_dir):
            print("Adding", split_dir)
            # 修改为直接使用split_dir参数，不再拼接DanceTrack前缀
            full_path = os.path.join(self.mot_path, split_dir)
            for vid in os.listdir(full_path):
                if 'seqmap' == vid:
                    continue
                # 修改这里：不要在vid前面加上split_dir
                # vid = os.path.join(split_dir, vid)  # 错误的路径拼接
                vid_path = os.path.join(split_dir, vid)  # 保持相对路径用于索引
                if 'DPM' in vid or 'FRCNN' in vid:
                    print(f'filter {vid}')
                    continue
                # 修改gt_path的构建方式
                gt_path = os.path.join(self.mot_path, split_dir, vid, 'gt', 'gt.txt')
                for l in open(gt_path):
                    t, i, *xywh, mark, label = l.strip().split(',')[:8]
                    t, i, mark, label = map(int, (t, i, mark, label))
                    if mark == 0:
                        continue
                    if label in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                        continue
                    else:
                        crowd = False
                    x, y, w, h = map(float, (xywh))
                    # 使用vid_path作为键
                    self.labels_full[vid_path][t].append([x, y, w, h, i, crowd])
    
        # 修改调用参数，直接使用"train"
        add_mot_folder("train")  # 不要使用 "DanceTrack/train"
        vid_files = list(self.labels_full.keys())
    
        self.indices = []
        self.vid_tmax = {}
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))
        print(f"Found {len(vid_files)} videos, {len(self.indices)} frames")
    
        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        self.period_idx = 0
    
        # 移除crowdhuman相关代码
        self.ch_indices = []
        
        # 修改检测数据库加载逻辑
        if args.det_db:
            det_db_path = args.det_db
            if os.path.exists(det_db_path):
                print(f"加载检测数据库: {det_db_path}")
                try:
                    with open(det_db_path) as f:
                        det_db = json.load(f)
                        # 将普通字典转换为defaultdict
                        self.det_db = defaultdict(list, det_db)
                        print(f"成功加载检测数据库，包含 {len(self.det_db)} 个条目")
                except Exception as e:
                    print(f"加载检测数据库失败: {e}")
                    self.det_db = defaultdict(list)
            else:
                print(f"警告: 检测数据库文件 {det_db_path} 不存在，使用空字典代替")
                self.det_db = defaultdict(list)
        else:
            self.det_db = defaultdict(list)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes'][:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances

    # 移除或注释掉load_crowd方法
    # def load_crowd(self, index):
    #     ID, boxes = self.ch_indices[index]
    #     boxes = copy.deepcopy(boxes)
    #     img = Image.open(self.ch_dir / 'Images' / f'{ID}.jpg')
    
    #     w, h = img._size
    #     n_gts = len(boxes)
    #     scores = [0. for _ in range(len(boxes))]
    #     for line in self.det_db[f'crowdhuman/train_image/{ID}.txt']:
    #         *box, s = map(float, line.split(','))
    #         boxes.append(box)
    #         scores.append(s)
    #     boxes = torch.tensor(boxes, dtype=torch.float32)
    #     areas = boxes[..., 2:].prod(-1)
    #     boxes[:, 2:] += boxes[:, :2]
    
    #     target = {
    #         'boxes': boxes,
    #         'scores': torch.as_tensor(scores),
    #         'labels': torch.zeros((n_gts, ), dtype=torch.long),
    #         'iscrowd': torch.zeros((n_gts, ), dtype=torch.bool),
    #         'image_id': torch.tensor([0]),
    #         'area': areas,
    #         'obj_ids': torch.arange(n_gts),
    #         'size': torch.as_tensor([h, w]),
    #         'orig_size': torch.as_tensor([h, w]),
    #         'dataset': "CrowdHuman",
    #     }
    #     rs = T.FixedMotRandomShift(self.num_frames_per_batch)
    #     return rs([img], [target])

    def _pre_single_frame(self, vid, idx: int):
        # 修改img_path的构建方式
        vid_parts = vid.split('/')
        # img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:08d}.jpg')
                # 修改：尝试不同的图片文件名格式
        img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx}.jpg')
        if not os.path.exists(img_path):
            # 如果没有找到不带前导零的文件，尝试带前导零的格式
            img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:08d}.jpg')

        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        obj_idx_offset = self.video_dict[vid] * 100000  # 100000 unique ids is enough for a video.

        targets['dataset'] = 'MOT17'
        targets['boxes'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['scores'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        for *xywh, id, crowd in self.labels_full[vid][idx]:
            targets['boxes'].append(xywh)
            assert not crowd
            targets['iscrowd'].append(crowd)
            targets['labels'].append(0)
            targets['obj_ids'].append(id + obj_idx_offset)
            targets['scores'].append(1.)
        # 修改检测数据库访问逻辑
        txt_key = os.path.join(vid, 'img1', f'{idx:08d}.txt')
        for line in self.det_db.get(txt_key, []):  # 使用get方法避免KeyError
            try:
                *box, s = map(float, line.split(','))
                targets['boxes'].append(box)
                targets['scores'].append(s)
            except Exception as e:
                print(f"处理检测结果时出错: {e}, line: {line}")
                continue

        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float64)
        targets['scores'] = torch.as_tensor(targets['scores'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 2:] += targets['boxes'][:, :2]
        return img, targets

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, vid, indices):
        return zip(*[self._pre_single_frame(vid, i) for i in indices])

    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        rate = randint(1, self.sample_interval + 1)
        tmax = self.vid_tmax[vid]
        ids = [f_index + rate * i for i in range(self.num_frames_per_batch)]
        return [min(i, tmax) for i in ids]

    def __getitem__(self, idx):
        # ... 现有代码 ...
        vid, frame_id = self.indices[idx]
        indices = self.sample_indices(vid, frame_id)
        images, targets = self.pre_continuous_frames(vid, indices)
        
        if self.transform is not None:
            images, targets = self.transform(images, targets)
        gt_instances, proposals = [], []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
            n_gt = len(targets_i['labels'])
            proposals.append(torch.cat([
                targets_i['boxes'][n_gt:],
                targets_i['scores'][n_gt:, None],
            ], dim=1))
        
        # 修复：保持proposals作为列表，不要合并为单个张量
        # 确保每个元素都是张量，而不是将整个列表合并为一个张量
        if len(proposals) == 0:
            # 如果没有proposals，添加一个空张量到列表中
            proposals = [torch.zeros((0, 5), dtype=torch.float32)]

        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'proposals': proposals,  # 保持为列表形式
        }

    def __len__(self):
        # 移除ch_indices的长度计算
        return len(self.indices)


class DetMOTDetectionValidation(DetMOTDetection):
    def __init__(self, args, seqs_folder, transform):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, transform)


def make_transforms_for_mot17(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([800, 1000, 1200]),
                    T.FixedMotRandomCrop(800, 1200),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            T.MOTHSV(),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_transform(args, image_set):
    mot17_train = make_transforms_for_mot17('train', args)
    mot17_test = make_transforms_for_mot17('val', args)

    if image_set == 'train':
        return mot17_train
    elif image_set == 'val':
        return mot17_test
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    # 添加详细路径验证逻辑
    if not root.exists():
        raise FileNotFoundError(f"数据集路径不存在，请检查配置：{root.absolute()}")
    
    # 验证子目录结构
    required_dirs = ['train', 'val', 'test']
    missing_dirs = [d for d in required_dirs if not (root / d).exists()]
    if missing_dirs:
        raise FileNotFoundError(f"数据集目录缺少以下必需子目录：{missing_dirs}")

    transform = build_transform(args, image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=str(root), transform=transform)
    elif image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=str(root), transform=transform)
    return dataset
