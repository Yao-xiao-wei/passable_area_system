import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_learning.datasets import PassableDataset
from deep_learning.model.unet import UNet

# =====================
# 基础配置
# =====================
BASE_DIR = r"D:\passable_area_system"
MODEL_DIR = os.path.join(BASE_DIR, "deep_learning", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

image_dir = os.path.join(BASE_DIR, "data", "raw")
mask_dir = os.path.join(BASE_DIR, "data", "masks")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Dice Loss（标准 batch 版本）
# =====================
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))

    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice

    return loss.mean()

# =====================
# 训练函数
# =====================
def train():

    dataset = PassableDataset(image_dir, mask_dir, img_ext="jpg")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet().to(device)

    # 🔴 关键：类别不平衡修正
    pos_weight = torch.tensor([3.0]).to(device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # 🔴 自动学习率衰减
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,

    )

    for epoch in range(40):

        model.train()
        total_loss = 0
        total_bce = 0
        total_dice = 0

        for img, mask in loader:

            img = img.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            logits = model(img)

            bce = criterion_bce(logits, mask)

            probs = torch.sigmoid(logits)
            dice = dice_loss(probs, mask)

            loss = bce + dice

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_dice += dice.item()

        avg_loss = total_loss / len(loader)
        avg_bce = total_bce / len(loader)
        avg_dice = total_dice / len(loader)

        scheduler.step(avg_loss)

        print(f"Epoch [{epoch+1}/40] "
              f"Loss={avg_loss:.4f} "
              f"BCE={avg_bce:.4f} "
              f"Dice={avg_dice:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "unet_v4.pth"))
    print("Model saved: unet_v4.pth")


if __name__ == "__main__":
    train()