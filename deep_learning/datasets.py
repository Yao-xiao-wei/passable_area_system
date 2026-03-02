import os
import glob
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset

class PassableDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_ext="jpg", mask_ext="png"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext

        self.images = []

        # 获取所有图像文件
        all_images = glob.glob(os.path.join(image_dir, f"*.{img_ext}"))
        print(f"Found {len(all_images)} images in {image_dir}")  # 打印图像文件数

        # 仅选择有对应标签的图像
        for img_path in all_images:
            mask_path = os.path.join(mask_dir, os.path.basename(img_path).replace(img_ext, mask_ext))

            # 打印图像和标签路径，检查是否匹配
            print(f"Checking: Image: {img_path} -> Mask: {mask_path}")  # 打印路径

            # 检查标签文件是否存在
            if os.path.exists(mask_path):
                self.images.append(img_path)
            else:
                print(f"Warning: No mask found for {img_path}")  # 警告没有标签

        if len(self.images) == 0:
            raise RuntimeError("No valid image-mask pairs found.")
        else:
            print(f"Found {len(self.images)} valid image-mask pairs.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path).replace(self.img_ext, self.mask_ext))

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 读取 mask
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # 数据增强（例如随机水平翻转）
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # 归一化和 resize
        img = cv2.resize(img, (256, 256)) / 255.0
        mask = cv2.resize(mask, (256, 256))

        # 转为 torch tensor
        img = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask