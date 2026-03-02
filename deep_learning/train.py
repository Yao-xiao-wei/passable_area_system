import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_learning.datasets import PassableDataset
from deep_learning.model.unet import UNet
import os

# 配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = r"D:\passable_area_system\deep_learning\model"
os.makedirs(MODEL_DIR, exist_ok=True)


# Dice Loss 计算
def dice_loss(pred, target, smooth=1e-6):
    intersection = torch.sum(pred * target)
    return 1 - (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)


def train():
    # 加载数据集
    image_dir = os.path.join(BASE_DIR, "data", "raw")
    mask_dir = os.path.join(BASE_DIR, "data", "masks")
    dataset = PassableDataset(image_dir, mask_dir, img_ext="jpg")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型初始化
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_bce = nn.BCEWithLogitsLoss()

    # 训练循环
    for epoch in range(40):
        total_loss = 0
        total_dice = 0

        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            # 前向传播
            pred = model(img)

            # 计算 BCE Loss
            bce_loss = criterion_bce(pred, mask)

            # 计算 Dice Loss
            pred_sigmoid = torch.sigmoid(pred)
            dice = dice_loss(pred_sigmoid, mask)

            # 总损失 = BCE + Dice Loss
            loss = bce_loss + dice
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice.item()

        print(f"Epoch [{epoch + 1}/40], Loss: {total_loss:.4f}, Dice Loss: {total_dice:.4f}")

    # 保存模型
    model_path = os.path.join(MODEL_DIR, "unet.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存: {model_path}")


if __name__ == "__main__":
    train()