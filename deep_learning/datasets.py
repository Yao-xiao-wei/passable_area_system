import os
import cv2
import torch
from torch.utils.data import Dataset


class PassableDataset(Dataset):

    def __init__(self, image_dir, mask_dir, img_ext="jpg", size=256):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.img_ext = img_ext

        self.images = [
            f for f in os.listdir(image_dir)
            if f.endswith(img_ext)
        ]

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.size, self.size))

        # 读取mask
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.size, self.size))

        # 二值化
        mask = (mask > 127).astype("float32")

        # 归一化
        image = image / 255.0

        # tensor
        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask