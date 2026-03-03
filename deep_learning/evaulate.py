import os
import cv2
import torch
import numpy as np
from deep_learning.model.unet import UNet

BASE_DIR = r"D:\passable_area_system"
IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
MODEL_PATH = os.path.join(BASE_DIR, "deep_learning", "model", "unet_v4.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():

    image_list = []
    for f in os.listdir(IMAGE_DIR):
        if f.endswith(".jpg"):
            mask_name = f.replace(".jpg", ".png")
            if os.path.exists(os.path.join(MASK_DIR, mask_name)):
                image_list.append(f)

    print(f"Total evaluation samples: {len(image_list)}")

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0

    valid_foreground_samples = 0
    background_only_samples = 0

    for name in image_list:

        img_path = os.path.join(IMAGE_DIR, name)
        mask_path = os.path.join(MASK_DIR, name.replace(".jpg", ".png"))

        img = cv2.imread(img_path)
        gt = cv2.imread(mask_path, 0)

        if img is None or gt is None:
            continue

        gt = (gt > 127).astype(np.uint8)

        # 判断是否为全黑GT
        if np.sum(gt) == 0:
            background_only_samples += 1
            continue

        valid_foreground_samples += 1

        h, w = gt.shape

        img_resized = cv2.resize(img, (256, 256)) / 255.0
        img_tensor = torch.tensor(
            np.transpose(img_resized, (2, 0, 1)),
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred = (probs > 0.5).astype(np.uint8)
        pred = cv2.resize(pred, (w, h))

        total_TP += np.sum((pred == 1) & (gt == 1))
        total_FP += np.sum((pred == 1) & (gt == 0))
        total_FN += np.sum((pred == 0) & (gt == 1))
        total_TN += np.sum((pred == 0) & (gt == 0))

    print("\n=== Dataset Composition ===")
    print("Foreground samples:", valid_foreground_samples)
    print("Background-only samples:", background_only_samples)

    if valid_foreground_samples == 0:
        print("\n⚠️ No foreground samples found in test set.")
        print("Cannot compute segmentation metrics.")
        return

    precision = total_TP / (total_TP + total_FP + 1e-6)
    recall = total_TP / (total_TP + total_FN + 1e-6)
    iou = total_TP / (total_TP + total_FP + total_FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    acc = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN + 1e-6)

    print("\n========= Evaluation Result =========")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"IoU      : {iou:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Accuracy : {acc:.4f}")


if __name__ == "__main__":
    evaluate()