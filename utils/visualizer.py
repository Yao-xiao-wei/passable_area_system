import cv2

def show_resizable(name, image):
    """
    可缩放窗口显示图像（滑块控制缩放比例）
    """
    h0, w0 = image.shape[:2]

    window_name = name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)

    def update_display(scale_val):
        new_w = max(1, int(w0 * scale_val / 100))
        new_h = max(1, int(h0 * scale_val / 100))
        resized = cv2.resize(image, (new_w, new_h))
        cv2.imshow(window_name, resized)

    cv2.createTrackbar("Scale %", window_name, 100, 500, lambda x: update_display(x))
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
