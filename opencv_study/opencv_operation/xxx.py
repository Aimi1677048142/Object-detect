import cv2
import numpy as np


def rotate_image_and_bboxes(image, bboxes, angle, scale=1.0):
    """
    旋转图像并调整边界框
    :param image: 输入图像
    :param bboxes: 边界框列表，格式为 [[x_min, y_min, x_max, y_max], ...]
    :param angle: 旋转角度，正值表示逆时针旋转，负值表示顺时针旋转
    :param scale: 缩放比例，默认为1.0
    :return: 旋转后的图像和调整后的边界框
    """
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 生成旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 旋转图像
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # 旋转边界框
    rotated_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox

        # 获取边界框的四个角点
        corners = np.array([
            [x_min, y_min],  # 左上角
            [x_max, y_min],  # 右上角
            [x_min, y_max],  # 左下角
            [x_max, y_max]  # 右下角
        ])

        # 为角点添加第三列，用于矩阵乘法的仿射变换
        ones = np.ones((4, 1))
        corners = np.hstack([corners, ones])

        # 旋转角点
        rotated_corners = M.dot(corners.T).T

        # 计算旋转后边界框的新坐标
        x_min_rotated = max(0, np.min(rotated_corners[:, 0]))
        y_min_rotated = max(0, np.min(rotated_corners[:, 1]))
        x_max_rotated = min(w, np.max(rotated_corners[:, 0]))
        y_max_rotated = min(h, np.max(rotated_corners[:, 1]))

        rotated_bboxes.append([x_min_rotated, y_min_rotated, x_max_rotated, y_max_rotated])

    return rotated_image, np.array(rotated_bboxes)


# 示例使用
image = cv2.imread(r'D:\pcb_data\original_ img and yolo_imfo\original img\aimi.jpg')
image = cv2.resize(image, [512, 512])

# 假设边界框格式为 [[x_min, y_min, x_max, y_max], ...]
bboxes = np.array([[0, 0, 200, 200], [100, 100, 250, 250]])

# 设置旋转角度
angle = 45

# 旋转图像并调整边界框
rotated_image, rotated_bboxes = rotate_image_and_bboxes(image, bboxes, angle)

# 输出旋转后的边界框
print("旋转后的边界框坐标:", rotated_bboxes)


# 显示旋转后的图像和边界框
for bbox in rotated_bboxes:
    cv2.rectangle(rotated_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

cv2.imshow("Rotated Image", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
