from PIL import Image, ImageDraw
import json
import os
import re
import pdb
import numpy as np
from tqdm import tqdm
import random
import argparse
from IPython.display import display

prefix = "/home/qinghong/data/GUI_database"
miniwob_url = f"{prefix}/MiniWob/metadata/miniwob_data_train.json"
imgs_dir =  f"{prefix}/MiniWob/images"
meta_dir = f"{prefix}/MiniWob/metadata/"

miniwob = json.load(open(miniwob_url, 'r'))

def normalize_bbox(bbox, size):
    x1, y1, x2, y2 = bbox
    width, height = size
    
    x1_norm = x1 / width
    y1_norm = y1 / height
    x2_norm = x2 / width
    y2_norm = y2 / height
    return [x1_norm, y1_norm, x2_norm, y2_norm]

def draw_point_bbox(image_path, point=None, bbox=None, radius=5, line=3):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    if point is not None:
        x, y = point[0] * width, point[1] * height
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='blue', outline='blue')
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
        draw.rectangle([x1, y1, x2, y2], outline='red', width=line)

    img_draw = np.array(image)
    display(Image.fromarray(img_draw))
    return

total_step = []
step_i = 0
num_epi = 0
for scenario, scenario_data in miniwob.items():
    for episode in scenario_data:
        num_epi += 1
        step_history = []
        for i, step in enumerate(episode):
            filename = step['img_filename']
            img_path = os.path.join(imgs_dir, filename)
            goal = step['goal']
            
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path)

            if step['action_type'] == 'click':
                action_meta = normalize_bbox(step['bbox'], image.size)
                # draw_point_bbox(img_path, bbox=action_meta)
            elif step['action_type'] == 'type':
                action_meta = step['typed_text']
            else:
                print(step)
            tmp_step = {
                "img_url": filename,
                "action_type": step['action_type'],
                "action_meta": action_meta,
            }
            total_step.append({
                "split": scenario,
                "id": "miniwob_{}".format(step_i), 
            
                "task": goal,
                "img_url": filename,
                "img_size": image.size,
                
                "action_type": step['action_type'],
                "action_meta": action_meta,

                "step_id": i,
                "step": tmp_step,
                "step_history": step_history.copy(),
                })
    
            step_history.append(tmp_step)
    
            step_i += 1

save_url = f"{meta_dir}/hf_train.json"
with open(save_url, "w") as file:
    json.dump(total_step, file, indent=4)