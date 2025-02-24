import numpy as np


def yolo_center_x_center_y_to_xy(yolo_box):
    min_xy = yolo_box[..., 1:3] - yolo_box[..., 3:5] / 2
    max_xy = yolo_box[..., 1:3] + yolo_box[..., 3:5] / 2
    return np.concatenate([min_xy, max_xy], axis=1)


def iou(g_box, predict_box):
    g_box_xy = yolo_center_x_center_y_to_xy(g_box)
    predict_box_xy = yolo_center_x_center_y_to_xy(predict_box)
    min_x_array = np.minimum(g_box_xy[..., 0], predict_box_xy[..., 0])
    min_y_array = np.minimum(g_box_xy[..., 1], predict_box_xy[..., 1])
    max_x_array = np.maximum(g_box_xy[..., 2], predict_box_xy[..., 2])
    max_y_array = np.maximum(g_box_xy[..., 3], predict_box_xy[..., 3])
    g_box_area = g_box[..., 3] * g_box[..., 4]
    predict_box_area = predict_box[..., 3] * predict_box[..., 4]

    inter_area = np.maximum(0, max_x_array - min_x_array) * np.maximum(0, max_y_array - min_y_array)
    area = g_box_area + predict_box_area - inter_area
    return inter_area / (g_box_area + predict_box_area - inter_area)


box1 = np.array([[0, 0.630556, 0.388889, 0.350000, 0.500000]])
box2 = np.array([[0, 0.630556, 0.388889, 0.350000, 0.500000]])
print(iou(box1, box2))

