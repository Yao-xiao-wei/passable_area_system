import os
import cv2
import torch
import numpy as np
from deep_learning.model.unet import UNet

class DeepLearningSegmentor:
    def __init__(self, weight_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()
        print("模型权重加载成功！")

    def segment(self, img_path):
        """输入图片路径，返回原图和预测 mask (0/1, 与原图同尺寸)"""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        orig_h, orig_w = img.shape[:2]

        # 预处理
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (256, 256))  # UNet 输入大小
        input_img = input_img / 255.0  # 归一化到 [0, 1]

        input_tensor = torch.from_numpy(input_img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        print("Input tensor shape:", input_tensor.shape)
        print("Input tensor min/max:", input_tensor.min().item(), input_tensor.max().item())

        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = output[0, 0].cpu().numpy()  # 单通道输出
            pred_mask = (pred > 0.5).astype(np.uint8)  # 二值化处理

            print("Pred min/max:", pred.min(), pred.max())

        # 缩放回原图尺寸
        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return img, pred_mask

def visualize(img, mask, save_path=None, window_name=None, max_width=960, max_height=540):
    """可视化预测结果，窗口可缩放，支持保存原尺寸"""
    overlay = img.copy()
    overlay[mask == 1] = [0, 255, 0]  # 标记可通行区域为绿色
    combined = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # 保存原图 + mask
    if save_path:
        cv2.imwrite(save_path, combined)

    # 显示缩放图像
    if window_name:
        h, w = combined.shape[:2]
        scale_w = min(1.0, max_width / w)
        scale_h = min(1.0, max_height / h)
        scale = min(scale_w, scale_h)
        display_img = cv2.resize(combined, (int(w * scale), int(h * scale)))

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()