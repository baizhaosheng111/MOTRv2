#!/usr/bin/env python
# 修复det_db_anno202503.json的格式

import json
from pathlib import Path

# 读取原始JSON文件
input_file = '/home/xyj/MOTRv2/data/det_db_anno202503.json'
output_file = '/home/xyj/MOTRv2/data/det_db_anno202503_fixed.json'

with open(input_file, 'r') as f:
    det_db = json.load(f)

# 创建新的字典
new_det_db = {}

# 修改格式
for key, values in det_db.items():
    # 修改键名格式
    parts = key.split('/')
    if len(parts) >= 4:
        # 从train/321600891741096593742/img1/00000063.txt
        # 变为anno202503/train/321600891741096593742/img1/00000063.txt
        new_key = f"anno202503/{key}"
        
        # 修改值格式，添加换行符
        new_values = [f"{v}\n" for v in values]
        
        # 添加到新字典
        new_det_db[new_key] = new_values

# 保存修改后的JSON文件（修改此行）
with open(output_file, 'w', encoding='ascii') as f:
    json.dump(new_det_db, f, ensure_ascii=True, indent=None, separators=(',', ':'))

print(f"已将 {input_file} 修改为正确格式并保存至 {output_file}")