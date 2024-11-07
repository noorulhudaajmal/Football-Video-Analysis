import numpy as np


def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox    
    return int((x1+x2)/2), int((y1+y2)/2)


def measure_distance(bbox_pt1, bbox_pt2):
    return np.sqrt((bbox_pt1[0]-bbox_pt2[0])**2 + (bbox_pt1[1]-bbox_pt2[1])**2)


def get_bbox_bottom_center(bbox):
    x1, y1, x2, y2 = bbox    
    return int((x1+x2)/2), int(y2)