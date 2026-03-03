# utils/labelme_to_mask.py

import os
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def labelme_json_to_mask(json_dir, mask_dir, img_ext="png"):
    """
    将 Labelme 导出的 JSON 文件转换为二值 mask (0/1)
    
    Args:
        json_dir: JSON 文件所在目录
        mask_dir: 生成 mask 保存目录
        img_ext: mask 文件后缀，默认 png
    """
    os.makedirs(mask_dir, exist_ok=True)
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    
    if not json_files:
        print("未找到 JSON 文件，请确认目录")
        return

    for json_file in tqdm(json_files, desc="Converting JSON to mask"):
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 获取图像大小
        height = data["imageHeight"]
        width = data["imageWidth"]

        # 创建空白 mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # 遍历每个标注 shape
        for shape in data.get("shapes", []):
            points = np.array(shape["points"], dtype=np.int32)
            label = shape.get("label", "floor")  # 可以按 label 做分类
            cv2.fillPoly(mask, [points], 1)  # 前景值 = 1

        # 保存 mask
        mask_name = os.path.splitext(json_file)[0] + "." + img_ext
        mask_path = os.path.join(mask_dir, mask_name)
        # 转为 0-255 PNG 保存
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_img)

    print(f"生成完成，mask 保存在 {mask_dir}")


if __name__ == "__main__":
    # 示例路径，可修改为你的项目路径
    json_dir = r"D:\passable_area_system\data\labels"
    mask_dir = r"D:\passable_area_system\data\masks"
    labelme_json_to_mask(json_dir, mask_dir)