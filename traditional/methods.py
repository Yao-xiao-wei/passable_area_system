from .hsv_v1_plain import segment as hsv_plain 
from .hsv_v2_roi import segment as hsv_roi 
from .hsv_v3_geometry import segment as hsv_geo 

METHODS = { 
    "HSV": hsv_plain, 
    "HSV+ROI": hsv_roi, 
    "HSV+ROI+GEO": hsv_geo, 
}
