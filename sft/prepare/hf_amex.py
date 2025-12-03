import os
import cv2
import re
import pdb
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from data_utils import is_english_simple, bbox_2_point

def is_english_simple(text):
    try:
        text.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def extract_box(box_str):
    box_str = box_str.replace("<box>", "").replace("</box>", "")
    x1, y1, x2, y2 = map(float, box_str.split(","))
    return x1, y1, x2, y2

def exclude_duplicate_question(ele):
    # we use this function to remove unwanted items
    question_list = ele["xml_desc"]
    question_set = set(question_list)
    if len(question_set) == 1:
        return question_set.pop()
    else:
        return None

def normalize_bbox(bbox, size):
    x1, y1, x2, y2 = bbox
    width, height = size
    
    x1_norm = x1 / width
    y1_norm = y1 / height
    x2_norm = x2 / width
    y2_norm = y2 / height
    return [x1_norm, y1_norm, x2_norm, y2_norm]

dataset_dir = "/home/qinghong/data/GUI_database/AMEX/"

def main(split="name"):
    parser = argparse.ArgumentParser(description="Example of argparse usage.")
    parser.add_argument(
        "--web_imgs",
        default=f"{dataset_dir}/images",
        help="Path to the directory containing web images.",
    )
    element_anno_path = (
        f"{dataset_dir}/AMEX/element_anno"
    )
    args = parser.parse_args()
    meta_data_path_list = os.listdir(element_anno_path)

    annotation_json = dict()  # annotation_data: dict
    annotation_json["info"] = []
    annotation_json["licenses"] = []
    annotation_json["images"] = []
    annotation_json["annotations"] = []
    annotation_json["categories"] = [
        {"supercategory": "object", "id": 1, "name": "object"}
    ]

    Totol_box_num = 0
    count_remove_duplicate = 0
    anno_id = 0
    print("start iteration")

    screen_list = {}
    for img_id, json_path in enumerate(tqdm(meta_data_path_list)):
        json_file = json.load(open(os.path.join(element_anno_path, json_path)))
        clickable_elements = json_file["clickable_elements"]

        image_id = json_file["image_path"]

        img_filename = image_id
        img_path = os.path.join(args.web_imgs, img_filename)
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path)
        if image is None:
            continue

        # im_h, im_w = image.shape[:2]
        img_size = image.size
        im_w, im_h = image.size

        if img_filename in screen_list:
            print(f"Duplicate image: {img_filename}")

        img_caption = json_file.get("page_caption", None)
        if split == "cap":
            if img_caption is None:
                continue
            screen_list[img_filename] = dict(img_url=img_filename, img_size=img_size, 
                                caption=img_caption)
            continue

        element_list = []
        for ele in clickable_elements:
            if exclude_duplicate_question(ele) is None:
                count_remove_duplicate += 1
                continue
            else:
                ele_name = exclude_duplicate_question(ele)

            boxes = ele["bbox"]
            valid = True

            if any(coord < 0 for coord in boxes):
                valid = False
            # for x1, y1, x2, y2 in boxes:
            #     if any(coord < 0 for coord in [x1, y1, x2, y2]):
            #         valid = False
            #         break
            if len(ele_name) > 60 or ele_name.strip() == "":
                valid = False
            if "{" in ele_name or "}" in ele_name:
                valid = False
            if not is_english_simple(ele_name):
                valid = False
            if not valid:
                continue
            # print(boxes)

            bbox = normalize_bbox(boxes, img_size)
            if split == "ele":
                instruction = ele_name
            elif split == "func":
                instruction = ele.get('functionality', None)

            if instruction is None or instruction == "":
                continue

            element = {
                "img_filename": img_filename,
                "instruction": instruction,
                "bbox": bbox,
                "point": bbox_2_point(bbox),
            }
            element_list.append(element)
            Totol_box_num += 1

        if len(element_list) > 0:
            screen_list[img_filename] = dict(img_url=img_filename, img_size=img_size, 
                                        caption=img_caption, 
                                        element=element_list, element_size=len(element_list))

    data_list = []
    for img_id, img_filename in enumerate(screen_list):
        img_data = screen_list[img_filename]
        data_list.append(img_data)

    save_url = rf"{dataset_dir}/metadata/hf_train_{split}.json"
    os.makedirs(os.path.dirname(save_url), exist_ok=True)
    with open(save_url, "w") as f:
        json.dump(data_list, f, indent=4)
    
    print(split)
    print(f"Total number of images: {len(data_list)}")
    print(f"Total number of items: {Totol_box_num}")
    print(f"Total number of skipped: {count_remove_duplicate}")

if __name__ == "__main__":
    main("ele")
    main("func")
    main("cap")