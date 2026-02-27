import os
import glob
import json
import numpy as np
import cv2
import labelme


def convert_labelme_to_mask():

    input_dir = "data/labels"
    output_dir = "data/masks"

    os.makedirs(output_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    if len(json_files) == 0:
        print("没有找到 json 文件")
        return

    for json_file in json_files:

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        img = labelme.utils.img_b64_to_arr(data["imageData"])

        label_name_to_value = {
            "_background_": 0,
            "passable": 1
        }

        lbl, _ = labelme.utils.shapes_to_label(
            img.shape,
            data["shapes"],
            label_name_to_value
        )

        base_name = os.path.basename(json_file).replace(".json", ".png")
        save_path = os.path.join(output_dir, base_name)

        cv2.imwrite(save_path, lbl.astype(np.uint8))

        print(f"{base_name} 转换完成")

    print("全部转换完成")