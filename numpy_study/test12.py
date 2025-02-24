import numpy as np


def yolo_center_xy_to_xy(yolo_array):
    min_xy = yolo_array[..., 1:3] - yolo_array[..., 3:5] / 2
    max_xy = yolo_array[..., 1:3] + yolo_array[..., 3:5] / 2
    return np.concatenate([min_xy, max_xy], axis=1)


def iou(bbox, pre_bbox):
    area = bbox[..., 3] * bbox[..., 4] + pre_bbox[..., 3] * pre_bbox[..., 4]
    bbox = yolo_center_xy_to_xy(bbox)
    pre_bbox = yolo_center_xy_to_xy(pre_bbox)
    min_x_array = np.minimum(bbox[..., 0], pre_bbox[..., 0])
    min_y_array = np.minimum(bbox[..., 1], pre_bbox[..., 1])
    max_x_array = np.maximum(bbox[..., 2], pre_bbox[..., 2])
    max_y_array = np.maximum(bbox[..., 3], pre_bbox[..., 3])

    inter_area = np.maximum(0, max_x_array - min_x_array) * np.maximum(0, max_y_array - min_y_array)

    return inter_area / (area - inter_area)


box1 = np.array([[0, 0.630556, 0.388889, 0.350000, 0.500000]])
box2 = np.array([[0, 0.630556, 0.388889, 0.350000, 0.500000]])
print(iou(box1, box2))
