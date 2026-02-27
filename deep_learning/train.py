import torch
from torch.utils.data import DataLoader
from deep_learning.datasets import PassableDataset
from deep_learning.model.unet import UNet
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 绝对路径，确保目录存在
model_dir = r"D:\passable_area_system\deep_learning\model"
os.makedirs(model_dir, exist_ok=True)

# Dice Loss 计算
def dice_loss(pred, target, smooth=1e-6):
    intersection = torch.sum(pred * target)
    return 1 - (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)

def train():
    # 创建数据集，确保数据集路径正确
    dataset = PassableDataset(
        os.path.join(BASE_DIR, "data/raw"),  # 使用绝对路径
        os.path.join(BASE_DIR, "data/masks"),  # 使用绝对路径
        img_ext="jpg"
    )

    # 打印数据集大小
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 UNet 模型并将其移动到设备
    model = UNet().to(device)

    # 使用 Adam 优化器，并设置学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 使用 BCEWithLogitsLoss 作为损失函数（带 sigmoid）
    criterion_bce = torch.nn.BCEWithLogitsLoss()

    # 训练循环
    for epoch in range(40):
        total_loss = 0
        total_dice = 0

        # 每次加载一个批次的数据
        for img, mask in loader:
            # 打印当前批次标签的唯一值
            print(f"Mask unique values: {torch.unique(mask)}")  # 确保标签值为0和1

            # 将数据移动到设备上（CPU或GPU）
            img = img.to(device)
            mask = mask.to(device)

            # 前向传播：计算模型的输出
            pred = model(img)

            # 计算 BCE Loss
            bce_loss = criterion_bce(pred, mask)

            # 计算 Dice Loss
            pred_sigmoid = torch.sigmoid(pred)  # 如果是 BCEWithLogitsLoss, 要加 sigmoid
            dice = dice_loss(pred_sigmoid, mask)

            # 总损失 = BCE + Dice Loss
            loss = bce_loss + dice
            total_loss += loss.item()
            total_dice += dice.item()

            # 反向传播：更新模型参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每个epoch结束后打印总损失
        print(f"Epoch {epoch}, Loss {total_loss:.4f}, Dice Loss {total_dice:.4f}")

    # 保存训练好的模型权重
    # 保存模型
    model_path = os.path.join(model_dir, "unet.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存: {model_path}")

if __name__ == "__main__":
    train()