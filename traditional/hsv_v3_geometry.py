import cv2
import numpy as np


def keep_largest_bottom_component(mask: np.ndarray) -> np.ndarray:
    """
    保留：
    - 与图像底部相连
    - 面积最大的连通区域
    """
    h, w = mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return mask

    best_label = 0
    best_area = 0

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]

        # 是否接触图像底部
        touches_bottom = (y + bh) >= (h - 1)

        if touches_bottom and area > best_area:
            best_area = area
            best_label = label

    if best_label == 0:
        return np.zeros_like(mask)

    return (labels == best_label).astype(np.uint8)


def segment(image: np.ndarray) -> np.ndarray:
    """
    HSV + ROI + Geometry 地面可通行区域分割（最终冻结版）

    输入:
        image: RGB 图像 (H, W, 3)
    输出:
        mask: 0/1 二值图 (H, W)
    """

    h, w, _ = image.shape

    # ---------- 1. RGB → HSV ----------
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # ---------- 2. HSV 阈值（经验 + 可解释） ----------
    lower = np.array([0, 0, 60])
    upper = np.array([180, 80, 255])
    binary = cv2.inRange(hsv, lower, upper)

    # ---------- 3. ROI：只保留下半部分 ----------
    roi_mask = np.zeros_like(binary)
    roi_mask[int(h * 0.5):, :] = 255
    binary = cv2.bitwise_and(binary, roi_mask)

    # ---------- 4. 形态学去噪 ----------
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ---------- 5. 转为 0/1 ----------
    mask = (binary > 0).astype(np.uint8)

    # ---------- 6. 几何一致性约束 ----------
    mask = keep_largest_bottom_component(mask)

    return mask
