from .hsv_v3_geometry import segment as hsv_segment

class TraditionalSegmentor:
    def segment(self, image):
        # 如果 hsv_segment 只返回 mask
        mask = hsv_segment(image)
        return mask
