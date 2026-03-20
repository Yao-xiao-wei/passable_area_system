import os
import cv2
import torch
import numpy as np
from torchvision import transforms

from model.unet import UNet


# =========================
# 路径配置
# =========================
MODEL_PATH = r"D:\passable_area_system\deep_learning\model\unet_best.pth"
IMAGE_DIR = r"D:\passable_area_system\data\raw"
SAVE_DIR = r"D:\passable_area_system\data\predictions"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 256


# =========================
# 模型加载
# =========================
def load_model():
    model = UNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# =========================
# 预处理
# =========================
def preprocess(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


# =========================
# 后处理（优化版）
# =========================
def postprocess(pred, original_shape):
    pred = pred.squeeze().cpu().numpy()

    # ⭐阈值优化（关键）
    mask = (pred > 0.3).astype(np.uint8) * 255

    # ⭐形态学优化（连通性）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 恢复尺寸
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

    return mask


# =========================
# 障碍检测
# =========================
def detect_obstacle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    obstacle = (edges > 0).astype(np.uint8) * 255
    return obstacle


# =========================
# 融合（优化版）
# =========================
def fuse(floor, obstacle):
    floor = floor / 255.0
    obstacle = obstacle / 255.0

    # ⭐避免一刀切
    passable = floor * (1 - 0.5 * obstacle)

    return (passable * 255).astype(np.uint8)


# =========================
# 可视化
# =========================
def visualize(image, mask):
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] = mask
    return cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)


# =========================
# 批量推理（最终版）
# =========================
def run_batch():
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = load_model()

    image_list = [f for f in os.listdir(IMAGE_DIR)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"共 {len(image_list)} 张图片")

    for idx, img_name in enumerate(image_list):
        img_path = os.path.join(IMAGE_DIR, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        input_tensor = preprocess(image)

        with torch.no_grad():
            probs = model(input_tensor)

        # =========================
        # 1️⃣ 纯模型结果
        # =========================
        floor_mask = postprocess(probs, (h, w))

        # =========================
        # 2️⃣ 融合结果
        # =========================
        obstacle_mask = detect_obstacle(image)
        passable_mask = fuse(floor_mask, obstacle_mask)

        # =========================
        # 保存（重点）
        # =========================
        cv2.imwrite(os.path.join(SAVE_DIR, "floor_" + img_name), floor_mask)
        cv2.imwrite(os.path.join(SAVE_DIR, "passable_" + img_name), passable_mask)

        # 可视化
        result = visualize(image, passable_mask)
        cv2.imwrite(os.path.join(SAVE_DIR, "vis_" + img_name), result)

        print(f"[{idx+1}/{len(image_list)}] {img_name}")

    print("✅ 推理完成")


if __name__ == "__main__":
    run_batch()