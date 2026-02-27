import glob
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PassableDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_ext="jpg"):
        # 绝对路径
        self.image_dir = os.path.join(BASE_DIR, image_dir)
        self.mask_dir = os.path.join(BASE_DIR, mask_dir)

        # 确保文件夹存在
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.images = []

        # 获取图片文件的路径
        pattern = os.path.join(self.image_dir, f"*.{img_ext}")
        all_images = sorted(glob.glob(pattern))

        missing_masks = []  # 用来存储缺少对应 mask 的图片

        # 自动过滤无对应mask的图片
        for img_path in all_images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.mask_dir, base_name + ".png")

            if os.path.exists(mask_path):
                self.images.append(img_path)
            else:
                missing_masks.append(img_path)  # 记录缺少 mask 的图片

        # 打印信息
        print(f"Found {len(all_images)} images")
        print(f"Valid image-mask pairs: {len(self.images)}")

        if missing_masks:
            print("Missing masks for the following images:")
            for missing in missing_masks:
                print(f"  - {missing}")

        if len(self.images) == 0:
            raise RuntimeError("No valid image-mask pairs found.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.resize(img, (256, 256))
        img = img / 255.0  # 归一化到 [0, 1]
        img = np.transpose(img, (2, 0, 1))  # 转换为 (C, H, W)

        # 读取 mask
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, base_name + ".png")

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = cv2.resize(mask, (256, 256))
        mask = np.expand_dims(mask, axis=0)  # 保持原始 0/1

        # 打印 mask 的唯一值，确保是 [0, 1]
        print(f"Mask unique values: {np.unique(mask)}")  # 应该是 [0 1]

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )