import cv2
import numpy as np

# =========================
# 读取图像
# =========================
img_path = r"D:\passable_area_system\data\raw\test.jpg"
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

h, w = image.shape[:2]

# =========================
# 创建窗口 & Trackbar
# =========================
cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)

def nothing(x):
    pass

# HSV 阈值
cv2.createTrackbar("H_min", "Tuner", 0, 180, nothing)
cv2.createTrackbar("H_max", "Tuner", 180, 180, nothing)
cv2.createTrackbar("S_min", "Tuner", 0, 255, nothing)
cv2.createTrackbar("S_max", "Tuner", 80, 255, nothing)
cv2.createTrackbar("V_min", "Tuner", 50, 255, nothing)
cv2.createTrackbar("V_max", "Tuner", 255, 255, nothing)

# ROI 起始比例（%）
cv2.createTrackbar("ROI_start_%", "Tuner", 50, 90, nothing)

# 几何约束参数
cv2.createTrackbar("Min_Row_Pixels", "Tuner", int(w * 0.2), w, nothing)

# =========================
# 主循环
# =========================
while True:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 读取参数
    h_min = cv2.getTrackbarPos("H_min", "Tuner")
    h_max = cv2.getTrackbarPos("H_max", "Tuner")
    s_min = cv2.getTrackbarPos("S_min", "Tuner")
    s_max = cv2.getTrackbarPos("S_max", "Tuner")
    v_min = cv2.getTrackbarPos("V_min", "Tuner")
    v_max = cv2.getTrackbarPos("V_max", "Tuner")

    roi_start_pct = cv2.getTrackbarPos("ROI_start_%", "Tuner")
    min_row_pixels = cv2.getTrackbarPos("Min_Row_Pixels", "Tuner")

    # HSV mask
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # =========================
    # ROI
    # =========================
    roi_mask = np.zeros_like(mask)
    roi_y = int(h * roi_start_pct / 100)
    roi_mask[roi_y:, :] = 1

    mask_roi = mask * roi_mask

    # =========================
    # 几何约束（行一致性）
    # =========================
    geo_mask = np.zeros_like(mask_roi)

    for y in range(roi_y, h):
        if np.count_nonzero(mask_roi[y]) > min_row_pixels:
            geo_mask[y] = mask_roi[y]

    # =========================
    # 可视化
    # =========================
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] = geo_mask  # 绿色表示可通行区域

    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

    # 拼图显示
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    geo_vis = cv2.cvtColor(geo_mask, cv2.COLOR_GRAY2BGR)

    top = np.hstack((image, mask_vis))
    bottom = np.hstack((geo_vis, overlay))
    vis = np.vstack((top, bottom))

    cv2.imshow("Tuner", vis)

    key = cv2.waitKey(30)
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
