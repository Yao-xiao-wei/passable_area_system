import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deep_learning.datasets import PassableDataset
from deep_learning.model.unet import UNet

# =====================
# 基础路径配置
# =====================

BASE_DIR = r"D:\passable_area_system"

TRAIN_IMG = os.path.join(BASE_DIR, "data", "train", "images")
TRAIN_MASK = os.path.join(BASE_DIR, "data", "train", "masks")

VAL_IMG = os.path.join(BASE_DIR, "data", "val", "images")
VAL_MASK = os.path.join(BASE_DIR, "data", "val", "masks")

MODEL_DIR = os.path.join(BASE_DIR, "deep_learning", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================
# 设备
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Dice Loss
# =====================

def dice_loss(pred, target, smooth=1e-6):

    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))

    dice = (2 * intersection + smooth) / (union + smooth)

    loss = 1 - dice

    return loss.mean()

# =====================
# 验证函数
# =====================

def validate(model, loader, criterion_bce):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for img, mask in loader:

            img = img.to(device)
            mask = mask.to(device)

            logits = model(img)

            bce = criterion_bce(logits, mask)

            probs = torch.sigmoid(logits)
            dice = dice_loss(probs, mask)

            loss = bce + dice

            total_loss += loss.item()

    return total_loss / len(loader)

# =====================
# 训练函数
# =====================

def train():

    # dataset
    train_dataset = PassableDataset(TRAIN_IMG, TRAIN_MASK, img_ext="png")
    val_dataset = PassableDataset(VAL_IMG, VAL_MASK, img_ext="png")

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # model
    model = UNet().to(device)

    # BCE loss
    pos_weight = torch.tensor([3.0]).to(device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")

    epochs = 40

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        total_bce = 0
        total_dice = 0

        for img, mask in train_loader:

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

        avg_loss = total_loss / len(train_loader)
        avg_bce = total_bce / len(train_loader)
        avg_dice = total_dice / len(train_loader)

        # validation
        val_loss = validate(model, val_loader, criterion_bce)

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"TrainLoss={avg_loss:.4f} "
            f"BCE={avg_bce:.4f} "
            f"Dice={avg_dice:.4f} "
            f"ValLoss={val_loss:.4f}"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_path = os.path.join(MODEL_DIR, "unet_best.pth")

            torch.save(model.state_dict(), save_path)

            print("Model saved:", save_path)

    # 保存最终模型
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, "unet_last.pth")
    )

    print("Training finished")

# =====================
# main
# =====================

if __name__ == "__main__":

    train()