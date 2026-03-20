import os
import cv2
import numpy as np


GT_DIR = r"D:\passable_area_system\data\masks"
PRED_DIR = r"D:\passable_area_system\data\predictions"


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 127).astype(np.uint8)


def compute_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union != 0 else 1.0


def compute_dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    return 2 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) != 0 else 1.0


def evaluate(prefix):
    files = [f for f in os.listdir(PRED_DIR) if f.startswith(prefix)]

    total_iou, total_dice, count = 0, 0, 0

    print(f"\n🔍 评估: {prefix}\n")

    for f in files:
        name = f.replace(prefix, "")
        pred_path = os.path.join(PRED_DIR, f)
        gt_path = os.path.join(GT_DIR, name)

        if not os.path.exists(gt_path):
            continue

        pred = load_mask(pred_path)
        gt = load_mask(gt_path)

        if pred is None or gt is None:
            continue

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        iou = compute_iou(pred, gt)
        dice = compute_dice(pred, gt)

        total_iou += iou
        total_dice += dice
        count += 1

    if count == 0:
        print("❌ 无数据")
        return

    print(f"平均 IoU : {total_iou / count:.4f}")
    print(f"平均 Dice: {total_dice / count:.4f}")


if __name__ == "__main__":
    evaluate("floor_")     # 模型能力
    evaluate("passable_")  # 融合结果