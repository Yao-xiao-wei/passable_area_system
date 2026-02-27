import os
import cv2
import torch
import numpy as np
from deep_learning.model.unet import UNet

# -------------------- 配置 --------------------
MODEL_PATH = r"D:\passable_area_system\deep_learning\model\unet.pth"
TEST_IMAGE_PATH = r"D:\passable_area_system\data\raw\test127.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 可视化函数 --------------------
def show_resizable(window_name, image, max_width=960):
    h, w = image.shape[:2]
    scale = min(1.0, max_width / w)
    image_resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------- 加载模型 --------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("模型权重加载完成")

# -------------------- 读取图片 --------------------
img = cv2.imread(TEST_IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"图片未找到: {TEST_IMAGE_PATH}")

img_resized = cv2.resize(img, (256, 256))
img_input = img_resized / 255.0
img_input = np.transpose(img_input, (2, 0, 1))  # HWC -> CHW
img_input = torch.tensor(img_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# -------------------- 模型预测 --------------------
with torch.no_grad():
    pred = model(img_input)  # 输出 (1,1,H,W)
    pred_mask = (pred[0,0] > 0.5).cpu().numpy().astype(np.uint8)  # 二值化

# -------------------- 叠加显示 --------------------
mask_colored = np.zeros_like(img_resized)
mask_colored[:,:,1] = pred_mask * 255  # 绿色显示 mask
overlay = cv2.addWeighted(img_resized, 0.7, mask_colored, 0.3, 0)

# -------------------- 显示结果 --------------------
show_resizable("预测结果", overlay)