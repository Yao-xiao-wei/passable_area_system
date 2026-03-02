import os
import torch
import numpy as np
import cv2
from deep_learning.model.unet import UNet

# 配置
MODEL_PATH = r"D:\passable_area_system\deep_learning\model\unet.pth"
RAW_IMAGE_DIR = r"D:\passable_area_system\data\raw"
MASK_IMAGE_DIR = r"D:\passable_area_system\data\masks"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 遍历 raw 文件夹中的所有图像
image_paths = [os.path.join(RAW_IMAGE_DIR, img) for img in os.listdir(RAW_IMAGE_DIR) if img.endswith(".jpg")]

for img_path in image_paths:
    # 读取输入图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        continue

    # 保存原始图像的尺寸
    original_size = img.shape[:2]

    # 图像预处理
    img_resized = cv2.resize(img, (256, 256))  # 修改为与模型输入一致
    img_input = img_resized / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))  # HWC -> CHW
    img_input = torch.tensor(img_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 模型预测
    with torch.no_grad():
        pred = model(img_input)  # 输出 (1, 1, H, W)
        pred_sigmoid = torch.sigmoid(pred[0, 0]).cpu().numpy()  # sigmoid后的概率值

        # 打印预测概率的范围，检查是否过低
        print(f"Predicted probability range: min={np.min(pred_sigmoid)}, max={np.max(pred_sigmoid)}")

        # 将概率值转换为 0-255 之间的整数，然后转为 uint8 类型
        pred_sigmoid = (pred_sigmoid * 255).astype(np.uint8)

        # 尝试手动设置阈值，例如 0.6 来替代 Otsu 方法
        threshold = 150  # 0.6 的阈值映射到 [0, 255] 范围内
        pred_mask = (pred_sigmoid > threshold).astype(np.uint8)

        # 查看预测图像和输出概率
        cv2.imshow("Predicted Sigmoid Output", pred_sigmoid)  # 查看原始概率图像
        cv2.waitKey(0)

    # 恢复预测图像尺寸到原始输入图像的尺寸
    pred_mask_resized = cv2.resize(pred_mask, (original_size[1], original_size[0]))  # 恢复到原始图像大小

    # 确保掩码的尺寸与图像大小一致
    mask_colored = np.zeros_like(img)
    mask_colored[:, :, 1] = pred_mask_resized * 255  # 绿色显示预测 mask

    # 可视化：叠加原图与预测图
    overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)

    # 显示结果
    cv2.imshow(f"Prediction - {os.path.basename(img_path)}", overlay)
    cv2.waitKey(0)

    # 保存结果
    result_path = os.path.join(MASK_IMAGE_DIR, f"pred_{os.path.basename(img_path)}")
    cv2.imwrite(result_path, overlay)
    print(f"Saved result to: {result_path}")

cv2.destroyAllWindows()