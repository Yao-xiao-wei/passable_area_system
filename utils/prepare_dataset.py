import os
import shutil
import cv2
import numpy as np

# 数据集路径
SOURCE_ROOT = r"D:\passable_area_system\data\Labelled_Test_Data"

# 输出路径
DATA_ROOT = r"D:\passable_area_system\data"

RAW_DIR = os.path.join(DATA_ROOT, "raw")
MASK_DIR = os.path.join(DATA_ROOT, "masks")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

index = 0

for folder in os.listdir(SOURCE_ROOT):

    folder_path = os.path.join(SOURCE_ROOT, folder)

    if not os.path.isdir(folder_path):
        continue

    # 遍历三个子文件夹
    for sub in os.listdir(folder_path):

        sub_path = os.path.join(folder_path, sub)

        if not os.path.isdir(sub_path):
            continue

        img_path = os.path.join(sub_path, "source_image.png")
        mask_path = os.path.join(sub_path, "driveable_ground.png")

        if os.path.exists(img_path) and os.path.exists(mask_path):

            filename = f"{index:04d}.png"

            raw_dst = os.path.join(RAW_DIR, filename)
            mask_dst = os.path.join(MASK_DIR, filename)

            # 复制原图
            shutil.copy(img_path, raw_dst)

            # 读取mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # 二值化
            binary = np.zeros_like(mask)
            binary[mask > 0] = 255

            cv2.imwrite(mask_dst, binary)

            index += 1

print("Dataset preparation completed!")
print("Total samples:", index)