import cv2 
import numpy as np 

def keep_largest_component(mask): 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8) 
    if num_labels <= 1: 
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) 
    
    
def segment(image, debug=False): 
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) 
    
    lower = np.array([0, 0, 60]) 
    upper = np.array([180, 80, 255]) 
    
    binary = cv2.inRange(hsv, lower, upper) 
    
    kernel = np.ones((5, 5), np.uint8) 
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) 
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) 
    
    mask = (binary > 0).astype(np.uint8) 
    mask = keep_largest_component(mask) 
    
    if debug: 
        return mask, binary 
    return mask
    