import cv2 

def load_image(path): 
    image = cv2.imread(path) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    return image
    