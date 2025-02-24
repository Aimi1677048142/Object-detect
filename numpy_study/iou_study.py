import numpy as np


def iou(current_box, box):
    """
    计算iou
    :param current_box:
    :param box:
    :return:
    """
    x1, y1, x2, y2 = current_box[:4]
    x1_, y1_, x2_, y2_ = box[:4]
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)
    # 计算重合的区域
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    # 计算联合区域
    union_area = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - inter_area
    return inter_area / union_area


def nms(array_box, target_confidence):
    """
    计算nms，非极大值抑制
    :param array_box: 当前的box
    :param target_confidence: 目标置信
    :return:
    """
    array_box = array_box[array_box[:, 4].argsort()[::-1]]
    keep_box = []
    while len(array_box) > 0:
        current_box = array_box[0]
        keep_box.append(current_box)
        # 计算当前的框与其他框的iou
        others_box = array_box[1:]
        iou_array = np.array([iou(current_box, x[:4]) for x in others_box])
        array_box = others_box[iou_array < target_confidence]
    return np.array(keep_box)


boxes = np.array([
    [100, 100, 210, 210, 0.9],
    [105, 105, 215, 215, 0.8],
    [50, 50, 150, 150, 0.7],
    [100, 100, 220, 220, 0.6]
])
iou_threshold = 0.5
final_boxes = nms(boxes, iou_threshold)
print(final_boxes)
