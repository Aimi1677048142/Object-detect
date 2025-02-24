import logging
import os
import random

import cv2
import numpy as np
from PIL import Image


def letter_box(img, size, boxes=None):
    """
        计算缩放比例
        :param size:
        :param img:
        :param boxes:
        :return:
        """
    target_h, target_w = size[0], size[1]
    img_h, img_w = img.shape[:2]
    scale_x = target_w / img_w
    scale_y = target_h / img_h
    scale_ratio = min(scale_x, scale_y)
    new_height, new_width = int(img_h * scale_ratio), int(img_w * scale_ratio)
    img_resize = cv2.resize(img, [new_width, new_height])
    # 创建目标大小的图像并填充背景
    new_image = create_background_image(target_h, target_w, img)

    x_offset = (target_w - new_width) // 2
    y_offset = (target_h - new_height) // 2
    new_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resize

    boxes[..., 1] = (boxes[..., 1] * scale_ratio + x_offset).astype(int)
    boxes[..., 2] = (boxes[..., 2] * scale_ratio + y_offset).astype(int)
    boxes[..., 3] = (boxes[..., 3] * scale_ratio + x_offset).astype(int)
    boxes[..., 4] = (boxes[..., 4] * scale_ratio + y_offset).astype(int)

    top, bottom, left, right = y_offset, target_h - (y_offset + new_height), x_offset, target_w - (x_offset + new_width)
    return new_image, scale_ratio, boxes, top, bottom, left, right


def create_background_image(target_h, target_w, img):
    """
    创建背景版
    :param target_h:
    :param target_w:
    :param img:
    :return:
    """
    if len(img.shape) == 2:  # 如果是灰度图像
        new_image = np.ones((target_h, target_w), dtype=np.uint8) * 114
    else:  # 如果是彩色图像
        new_image = np.ones((target_h, target_w, img.shape[2]), dtype=np.uint8) * 114
    return new_image


def yolo_box_to_original_image(new_image, boxes, ratio: float, top: int, left: int,
                               category_file_path: str, distance: int, font_scale: float, image_name,
                               save_draw_picture_path):
    """
    在原图像中画框
    :param new_image:
    :param boxes:
    :param ratio:
    :param top: 图片的上边界偏移量（通常是在缩放过程中添加的边界）
    :param left: 图片的左边界偏移量
    :param category_file_path: 类别文件路径，用于获取类别标签
    :param distance: 在图片上标注类别时，类别标签与边界框之间的距离
    :param font_scale: 字体大小比例，用于标注类别时控制字体的大小
    :param image_name: 图像名称，用于保存最终图像时命名
    :param save_draw_picture_path: 保存最终带有标注图像的路径
    :return:
    """
    boxes[..., 1] = (boxes[..., 1] - left) / ratio
    boxes[..., 2] = (boxes[..., 2] - top) / ratio
    boxes[..., 3] = (boxes[..., 3] - left) / ratio
    boxes[..., 4] = (boxes[..., 4] - top) / ratio

    draw_frame_on_picture(boxes.astype(int), new_image, get_image_category(category_file_path), distance, font_scale,
                          image_name, save_draw_picture_path)


def get_label_name(file_path: str) -> dict:
    """
    获取label的名称并且拼装成字典
    :param file_path: label标签的文件路径
    :return:
    """
    label_dict = {}
    if not os.path.exists(file_path):
        return label_dict
    with open(file_path, mode='r', encoding='utf-8') as file:
        label_dict = {index: value for index, value in enumerate(file.read().splitlines())}
    return label_dict


def get_image_category(file_path: str) -> dict:
    """
    获取图片的类别信息
    :param file_path:
    :return:
    """
    category = get_label_name(file_path)
    if not category:
        category = {x: f'category{x}' for x in range(10)}
    return category


def draw_frame_on_picture(array_list, img,
                          category_name: dict,
                          distance: int,
                          font_scale: float,
                          image_name: str,
                          save_draw_picture_path: str):
    """
    针对一张图片进行画框，贴标签
    :param image_name:
    :param array_list:
    :param img:
    :param category_name:
    :param distance:
    :param font_scale:
    :param save_draw_picture_path:
    :return:
    """
    for obj in array_list:
        img_rectangle = cv2.rectangle(img, (obj[1], obj[2]), (obj[3], obj[4]), color=(0, 0, 255),
                                      thickness=2)
        cv2.putText(img_rectangle, category_name.get(obj[0]) + ":" + str(obj[0]), (obj[1], obj[2] - distance),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 255), thickness=2)
        # 使用 logging.log 记录日志信息
    if save_draw_picture_path:
        cv2.imwrite(os.path.join(save_draw_picture_path, image_name), img)
        logging.log(logging.INFO, "文件：%s完成绘制", image_name)
    else:
        show(img, 'img')


def show(image, title, is_debug=True, is_scale=False):
    """
    cv2用来展示图片
    :param image:
    :param title:
    :param is_debug:
    :param is_scale:
    :return:
    """
    if is_scale:
        img_h, img_w = image.shape[0:2]
        print(img_h, img_w)
        image = cv2.resize(image, [int(img_h * 0.5), int(img_w * 0.5)])
    if is_debug:
        cv2.imshow(title, image)
        cv2.waitKey(0)


def yolo_x_center_y_center_x_y(yolo_array):
    """
    numpy将yolo转换为voc
    :param yolo_array:
    :return:
    """
    category = yolo_array[..., 0].reshape(-1, 1)
    x_y_min = yolo_array[..., 1:3] - yolo_array[..., 3:5] / 2
    x_y_max = yolo_array[..., 1:3] + yolo_array[..., 3:5] / 2
    return np.concatenate([category, x_y_min, x_y_max], axis=1)


def bgr_to_rgb(image, boxes):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), boxes


def bgr_to_gray(image, boxes):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), boxes


def cv_equalize_rgb_histogram(image, boxes):
    """
    全局直方图均衡化
    :param image:
    :param boxes:
    :return:
    """
    image_chanel = len(image.shape)
    if 2 == image_chanel:
        image = cv2.equalizeHist(image)
    elif 3 == image_chanel and image.shape[2] == 3:
        channels = cv2.split(image)
        equalize_channels = [cv2.equalizeHist(channel) for channel in channels]
        image = cv2.merge(equalize_channels)
    else:
        raise ValueError("The input image format is incorrect")
    return image, boxes


def cv_apply_clahe_to_rgb(image, clip_limit: float, title_grid_size: tuple, boxes):
    """
    区域直方图均衡化
    :param image:
    :param clip_limit:
    :param title_grid_size:
    :param boxes:
    :return:
    """
    image_chanel = len(image.shape)
    clahe = cv2.createCLAHE(clip_limit, title_grid_size)
    if 2 == image_chanel:
        image = clahe.apply(image)
    elif 3 == image_chanel and image.shape[2] == 3:
        channels = cv2.split(image)
        clahe_channels = [clahe.apply(channel) for channel in channels]
        image = cv2.merge(clahe_channels)
    else:
        raise ValueError("The input image format is incorrect")
    return image, boxes


def random_brightness_adjust(image, random_min: float, random_max: float, boxes):
    """
    随机调整亮度
    :param image:
    :param random_min:
    :param random_max:
    :param boxes:
    :return:
    """
    random_ratio = random.uniform(random_min, random_max)
    # 使用 alpha 调整亮度，beta 设为 0 可以保持对比度不变，仅调整亮度
    return cv2.convertScaleAbs(image, alpha=random_ratio, beta=0), boxes


def image_inversion(image, boxes):
    return cv2.bitwise_not(image), boxes


def image_mean(image, boxes):
    image_chanel = len(image.shape)
    if 2 == image_chanel:
        mean = np.mean(image)
        image = image - mean
    elif 3 == image_chanel and image.shape[2] == 3:
        mean = np.mean(image, axis=(0, 1))
        image = image - mean
    else:
        raise ValueError("The input image format is incorrect")
    return image, boxes


def mix_up(image1, boxes1, image2, boxes2, alpha, beta):
    """
    图像堆叠
    :param image1:
    :param boxes1:
    :param image2:
    :param boxes2:
    :param alpha:
    :param beta:
    :return:
    """
    return cv2.addWeighted(image1, alpha, image2, beta, 0), alpha * boxes1 + beta * boxes2


def image_mosic(image1, boxes1, image2, boxes2, image3, boxes3, image4, boxes4, target_size=None, num=2):
    if target_size is None:
        target_size = np.array([1024, 1024])
    size = (target_size / num).astype(int)
    new_image1, scale_ratio, boxes1, top, bottom, left, right = letter_box(image1, size, boxes1)
    new_image2, scale_ratio, boxes2, top, bottom, left, right = letter_box(image2, size, boxes2)
    new_image3, scale_ratio, boxes3, top, bottom, left, right = letter_box(image3, size, boxes3)
    new_image4, scale_ratio, boxes4, top, bottom, left, right = letter_box(image4, size, boxes4)
    h, w = size[:2]
    image_mosaic = np.zeros([num * h, num * w, 3], dtype=np.uint8)
    image_mosaic[:h, :w, :] = new_image1
    image_mosaic[:h, w:num * w, :] = new_image2
    image_mosaic[h:num * h, :w, :] = new_image3
    image_mosaic[h:num * h, w:num * w, :] = new_image4
    new_boxes_list = []
    # 更新边界框坐标并加入新的边界框列表
    if boxes1.size > 0:
        new_boxes_list.append(boxes1)
    if boxes2.size > 0:
        boxes2[..., [1, 3]] += w
        new_boxes_list.append(boxes2)
    if boxes3.size > 0:
        boxes3[..., [2, 4]] += h
        new_boxes_list.append(boxes3)
    if boxes4.size > 0:
        boxes4[..., [1, 3]] += w
        boxes4[..., [2, 4]] += h
        new_boxes_list.append(boxes4)
    new_boxes = np.array(new_boxes_list).reshape(-1, 5)
    return image_mosaic, new_boxes


def slide_crop(image, boxes, pixes_value: int, size):
    """
    滑动窗口裁剪函数
    :param image: 输入的原始图像
    :param boxes: 图像中的边界框，通常是对象的坐标
    :param pixes_value: 滑动窗口重叠的像素值
    :param size: 目标尺寸（用于 letter_box 函数）
    :return: 返回裁剪后的图像列表和更新的边界框列表
    """
    # 通过 letter_box 函数将图像调整到目标大小，并获取新的图像、新的边界框、缩放比例和填充信息
    new_image, scale_ratio, new_boxes, top, bottom, left, right = letter_box(image, size, boxes)
    image_h, image_w = new_image.shape[:2]
    h_size, w_size = image_h // 2, image_h // 2
    # 定义四个裁剪窗口的坐标，裁剪窗口通过像素值进行调整
    image1_coordinate = [0, 0, w_size + pixes_value / 2, h_size + pixes_value / 2]
    image2_coordinate = [w_size - pixes_value / 2, 0, image_w, h_size + pixes_value / 2]
    image3_coordinate = [0, h_size - pixes_value / 2, w_size + pixes_value / 2, image_h]
    image4_coordinate = [w_size - pixes_value / 2, h_size - pixes_value / 2, image_w, image_h]
    # 初始化存储裁剪图像和更新边界框的列表
    cropped_images = []
    update_boxes = [[], [], [], []]
    # 遍历四个裁剪窗口
    for index, cropped_image in enumerate([image1_coordinate, image2_coordinate, image3_coordinate, image4_coordinate]):
        # 获取当前裁剪窗口的坐标，并将坐标转换为整数
        x_min, y_min, x_max, y_max = map(int, cropped_image)
        # 将裁剪后的图像添加到列表中 注意这里需要copy一下，因为切片是浅拷贝
        cropped_image = new_image.copy()[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped_image)
        box_x_min, box_y_min, box_x_max, box_y_max = new_boxes[..., 1], new_boxes[..., 2], new_boxes[..., 3], new_boxes[
            ..., 4]
        # 检查哪些边界框在当前裁剪窗口内
        flag_box = (box_x_min >= x_min) & (box_y_min >= y_min) & (box_x_max <= x_max) & (box_y_max <= y_max)
        # 更新符合条件的边界框并减去 x_min 和 y_min
        if np.any(flag_box):  # 检查是否有符合条件的边界框
            selected_boxes = new_boxes[flag_box]  # 提取符合条件的边界框
            selected_boxes[..., 1] -= x_min  # 更新 x_min
            selected_boxes[..., 2] -= y_min  # 更新 y_min
            selected_boxes[..., 3] -= x_min  # 更新 x_max
            selected_boxes[..., 4] -= y_min  # 更新 y_max
            update_boxes[index] = selected_boxes  # 转换为列表并添加到更新的边界框列表中
        else:
            update_boxes[index] = np.array([])  # 如果没有符合条件的边界框，添加空列表
    return cropped_images, update_boxes


def slip_cutting(image, boxes, pixes_value: int, size, num: int):
    """
    滑动裁图->使用传递参数的滑动
    :param image:
    :param boxes:
    :param pixes_value:
    :param size:
    :param num:
    :return:
    """
    new_image, _, new_boxes, _, _, _, _ = letter_box(image, size, boxes)
    # 计算裁图的大小
    img_h, img_w = new_image.shape[:2]
    h, w = int(size[0] / num + pixes_value / 2), int(size[1] / num + pixes_value / 2)
    step = h - pixes_value
    # 新建列表，存储图片和对应的boxes
    result_image = []
    result_boxes = []
    for tmp_h in range(0, img_h - h + 1, step):
        for tmp_w in range(0, img_w - w + 1, step):
            cutting_image = new_image.copy()[tmp_h:tmp_h + h, tmp_w:tmp_w + w]
            result_image.append(cutting_image)
            # 计算交集
            inter_x_min = np.maximum(boxes[..., 1], tmp_w)
            inter_y_min = np.maximum(boxes[..., 2], tmp_h)
            inter_x_max = np.minimum(boxes[..., 3], tmp_w + w)
            inter_y_max = np.minimum(boxes[..., 4], tmp_h + h)
            # 判断哪些 box 有交集
            valid_mask = (inter_x_min < inter_x_max) & (inter_y_min < inter_y_max)
            if np.any(valid_mask):
                selected_boxes = boxes[valid_mask].copy()
                # 更新 box 的坐标，使其相对于裁剪后的图像
                selected_boxes[..., 1] = inter_x_min[valid_mask] - tmp_w
                selected_boxes[..., 2] = inter_y_min[valid_mask] - tmp_h
                selected_boxes[..., 3] = inter_x_max[valid_mask] - tmp_w
                selected_boxes[..., 4] = inter_y_max[valid_mask] - tmp_h
                # 将更新后的 box 添加到 result_boxes
                result_boxes.append(selected_boxes)

    return result_image, result_boxes


"""
    锐化
"""


def laplace_blur(image, boxes, ksize: int, sigma, alpha: float, beta: float):
    """
    拉普拉斯模糊
    :param image:
    :param boxes:
    :param ksize:
    :param sigma:
    :param alpha
    :param beta
    :return:
    """
    # 对图像进行高斯模糊
    gaussian_image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    # 应用拉普拉斯算子
    laplacian = cv2.Laplacian(gaussian_image, cv2.CV_64F)
    # 转换为8位图像
    laplacian = cv2.convertScaleAbs(laplacian)
    new_image = cv2.addWeighted(laplacian, alpha, image, beta, sigma)
    return new_image, boxes


def gaussian_blur(image, boxes, ksize, alpha, beta, sigma):
    """
    高斯模糊锐化
    :param image:
    :param boxes:
    :param ksize:
    :param alpha:
    :param beta:
    :param sigma:
    :return:
    """
    gaussian_image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    new_image = cv2.addWeighted(gaussian_image, alpha, image, beta, sigma)
    return new_image, boxes


def mean_blur(image, boxes, ksize, alpha, beta, sigma):
    """
    均值滤波，也叫均值模糊
    :param image:
    :param boxes:
    :param ksize:
    :param alpha:
    :param beta:
    :param sigma:
    :return:
    """
    blur_image = cv2.blur(image, (ksize, ksize), sigma)
    new_image = cv2.addWeighted(blur_image, alpha, image, beta, sigma)
    return new_image, boxes


def edge_enhancement(image, boxes, ksize, alpha, beta):
    """
    使用Sobel算子进行边缘增强并将其重绘到原图上
    :param image: 输入的彩色图像
    :param boxes: 边界框（未修改）
    :param ksize: Sobel算子的核大小
    :param alpha: 原图像权重
    :param beta: 边缘图像权重
    :return: 增强后的图像和边界框
    """
    # 分离图像的三个通道
    channels = cv2.split(image)

    # 初始化一个列表来保存增强后的各通道图像
    enhanced_channels = []

    for channel in channels:
        # 将每个通道图像转换为灰度图
        image_gray = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

        # 进行Sobel算子计算x，y方向的梯度
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize)

        # 计算边缘梯度
        magnitude = cv2.magnitude(sobel_x, sobel_y)

        # 归一化
        enhanced_channel = np.uint8(np.clip(magnitude, 0, 255))

        # 将增强的通道图像与原图像的通道合并
        enhanced_channel_colored = cv2.addWeighted(channel, alpha, enhanced_channel, beta, 0)
        enhanced_channels.append(enhanced_channel_colored)

    # 合并所有通道图像
    enhanced_image = cv2.merge(enhanced_channels)

    return enhanced_image, boxes


def geometric_transformation_flip(image, boxes, flip_value: int):
    """
    几何变换-水平/垂直/水平垂直同时翻转
    :param image:
    :param boxes:
    :param flip_value:0为垂直翻转，1为水平翻转，-1为同时翻转
    :return:
    """
    image_h, image_w = image.shape[:2]
    image_flip = cv2.flip(image, flip_value)
    if 0 == flip_value:
        boxes[..., [2, 4]] = image_h - boxes[..., [2, 4]]
    elif 1 == flip_value:
        boxes[..., [1, 3]] = image_w - boxes[..., [2, 4]]
    else:
        boxes[..., [2, 4]] = image_h - boxes[..., [2, 4]]
        boxes[..., [1, 3]] = image_w - boxes[..., [2, 4]]
    return image_flip, boxes


def random_translate_image(image, boxes, tx_range, ty_range):
    """
    指定范围进行随机平移
    :param image:
    :param boxes:
    :param tx_range:
    :param ty_range:
    :return:
    """
    image_h, image_w = image.shape[:2]
    tx = random.randrange(tx_range[0], tx_range[1])
    ty = random.randrange(ty_range[0], ty_range[1])
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    translate_image = cv2.warpAffine(image, m, (image_h, image_w))
    boxes[..., [1, 3]] += tx
    boxes[..., [2, 4]] += ty

    return translate_image, boxes


def copy_boxes_to_image(image, specified_box, copy_num: int, iou_value: float):
    """
    将指定的边界框 (specified_box) 复制 n 份到图片中
    :param image: 输入的图像
    :param specified_box: 需要复制的目标边界框 (格式为 [class, x_min, y_min, x_max, y_max])
    :param copy_num: 要复制的边界框数量
    :param iou_value: 指定的 IOU 阈值，复制的边界框与其他框的重叠比例不能超过该值
    :return: 返回原始图像以及包含复制边界框的数组
    """
    # 切图
    image_h, image_w = image.shape[:2]
    x_min, y_min, x_max, y_max = specified_box[..., 1], specified_box[..., 2], specified_box[..., 3], specified_box[
        ..., 4]
    box_h, box_w = y_max - y_min, x_max - x_min
    # 初始化结果框列表，包含原始指定的边界框
    result_box = [specified_box]
    label_image = image.copy()[int(y_min):int(y_max), int(x_min):int(x_max)]
    i = 0
    # 复制循环，直到达到指定的复制数量
    while i < copy_num:
        new_x_min, new_y_min = random.randint(0, int(image_w - box_w)), random.randint(0, int(image_h - box_h))
        new_x_max, new_y_max = new_x_min + box_w, new_y_min + box_h
        predict_box = np.array([specified_box[..., 0], new_x_min, new_y_min, new_x_max, new_y_max]).astype(int)
        # 计算新边界框与结果框列表中所有边界框的 IOU（交并比），并存储结果
        array_iou = np.array([iou(box, predict_box) for box in result_box])
        # 如果新边界框与现有的所有边界框的 IOU 值都小于指定的阈值，则接受该边界框
        if np.all(array_iou < iou_value):
            # 将新的预测边界框添加到结果列表中
            image[predict_box[2]:predict_box[4], predict_box[1]:predict_box[3]] = label_image
            result_box.append(predict_box)
            i += 1
    return image, np.array(result_box)


def copy_boxes_to_image_by_mask(image, specified_box, copy_num: int):
    image_h, image_w = image.shape[:2]
    # 创建一个mask掩码
    mask = np.zeros([image_h, image_w], dtype=np.uint8)
    x_min, y_min, x_max, y_max = specified_box[..., 1], specified_box[..., 2], specified_box[..., 3], specified_box[
        ..., 4]
    box_h, box_w = y_max - y_min, x_max - x_min
    result_box = [specified_box]
    label_image = image.copy()[y_min:y_max, x_min:x_max]
    mask[y_min:y_max, x_min:x_max] = 1
    i = 0
    # 复制循环，直到达到指定的复制数量
    while i < copy_num:
        new_x_min, new_y_min = random.randint(0, int(image_w - box_w)), random.randint(0, int(image_h - box_h))
        new_x_max, new_y_max = new_x_min + box_w, new_y_min + box_h
        predict_box = np.array([specified_box[..., 0], new_x_min, new_y_min, new_x_max, new_y_max]).astype(int)
        if np.all(mask[new_y_min:new_y_max, new_x_min:new_x_max] == 0):
            mask[new_y_min:new_y_max, new_x_min:new_x_max] = 1
            image[new_y_min:new_y_max, new_x_min:new_x_max] = label_image
            result_box.append(predict_box)
            i += 1
    return image, np.array(result_box)


def iou(g_box, predict_box):
    """
    计算iou
    :param g_box:
    :param predict_box:
    :return:
    """
    min_x_array = np.maximum(g_box[..., 1], predict_box[..., 1])
    min_y_array = np.maximum(g_box[..., 2], predict_box[..., 2])
    max_x_array = np.minimum(g_box[..., 3], predict_box[..., 3])
    max_y_array = np.minimum(g_box[..., 4], predict_box[..., 4])
    g_box_area = (g_box[..., 4] - g_box[..., 2]) * (g_box[..., 3] - g_box[..., 1])
    predict_box_area = (predict_box[..., 4] - predict_box[..., 2]) * (predict_box[..., 3] - predict_box[..., 1])

    inter_area = np.maximum(0, max_x_array - min_x_array) * np.maximum(0, max_y_array - min_y_array)
    return inter_area / (g_box_area + predict_box_area - inter_area)


def replace_picture(image1, image2, size, flag, alpha: float, beta: float):
    """
    随机替换图片区域
    :param image1: 原始图像
    :param image2: 替换区域来源的图像
    :param size: 替换区域的大小（正方形区域的边长）
    :param flag: 控制是否进行图像加权融合，如果为 False，直接替换图片
    :param alpha: 用于加权融合时的第一个图像的权重
    :param beta: 用于加权融合时的第二个图像的权重
    :return: 返回替换或融合后的图像
    """
    image1_h, image1_w = image1.shape[:2]
    image1_h_index, image1_w_index = random.randint(0, image1_h), random.randint(0, image1_w)
    image2_h, image2_w = image2.shape[:2]
    image2_h_index, image2_w_index = random.randint(0, image2_h), random.randint(0, image2_w)
    if (image1_h_index + size > image2_h
            or image1_w_index + size > image1_w
            or image2_h_index + size > image2_h
            or image2_w_index + size > image2_w):
        return image1
    cropped_image = image2[image2_h_index:size + image2_h_index, image2_w_index:size + image2_w_index]
    image_new = cv2.addWeighted(
        image1[image1_h_index:size + image1_h_index, image1_w_index:size + image1_w_index],
        alpha, cropped_image, beta, 0)
    image1[image1_h_index:size + image1_h_index, image1_w_index:size + image1_w_index] = image_new
    if not flag:
        image1[image1_h_index:size + image1_h_index, image1_w_index:size + image1_w_index] = cropped_image
    return image1


def voc_to_yolo_array(img, voc_array):
    """
    voc数据转换为yolo数据numpy
    :param img:
    :param voc_array:
    :return:
    """
    image_h, image_w = img.shape[:2]
    category_array = voc_array[..., 0].reshape(-1, 1)
    x_center_array = ((voc_array[..., 1] + voc_array[..., 3]) / 2) / image_w
    y_center_array = ((voc_array[..., 2] + voc_array[..., 4]) / 2) / image_h
    width = (voc_array[..., 3] - voc_array[..., 1]) / image_w
    height = (voc_array[..., 4] - voc_array[..., 2]) / image_h
    return img, np.concatenate(
        [category_array, x_center_array.reshape(-1, 1), y_center_array.reshape(-1, 1), width.reshape(-1, 1),
         height.reshape(-1, 1)], axis=1)


def split_image_list(lst, group_size=4):
    """
    将列表按指定大小分割，并在最后一组不足时用第一个子列表的元素填充。
    支持 lst 中的元素为元组，并确保填充时 ndarray 的独立性。

    :param lst: 需要分割的列表，元素为 (ndarray, ndarray) 的元组
    :param group_size: 每组的大小，默认为 4
    :return: 分割后的列表，包含等分的子列表
    """
    # 首先将列表按照每 group_size 个元素进行分割
    result = [lst[i:i + group_size] for i in range(0, len(lst), group_size)]

    # 如果最后一个子列表不足 group_size 个元素，进行填充
    if len(result[-1]) < group_size:
        missing = group_size - len(result[-1])  # 计算缺失的元素数量
        first_sublist = result[0]  # 获取第一个子列表的元素

        # 使用第一个子列表的元组中的 ndarray 副本来填充，防止共享内存
        fill_elements = []
        idx = 0
        while len(fill_elements) < missing:
            # 复制元组中的每个 ndarray，避免对原数组的修改
            array1_copy = np.copy(first_sublist[idx % len(first_sublist)][0])
            array2_copy = np.copy(first_sublist[idx % len(first_sublist)][1])
            fill_elements.append((array1_copy, array2_copy))
            idx += 1

        result[-1].extend(fill_elements)  # 填充到最后一个子列表中

    return result


def picture_up_down(file_path_txt: str, file_path_jpg: str, category_name: dict,
                    distance: int,
                    font_scale: float,
                    picture_suffix: str,
                    file_suffix: str,
                    save_draw_picture_path: str,
                    expansion_pixel_value: int,
                    sampled_freq: int,
                    target_size: int):
    """
    多尺度图像数据增强方法：通过对图像进行上下采样并处理对应的边界框，实现数据增强。

    :param file_path_txt: 存储 YOLO 标签文件的文件夹路径。
    :param file_path_jpg: 存储图像文件的文件夹路径。
    :param category_name: 类别名称的字典，用于标识图像中的物体类别。
    :param distance: 绘制框时的距离参数，用于调整框与文本的距离。
    :param font_scale: 绘制框时字体的缩放比例，用于调整标注文本的大小。
    :param picture_suffix: 图像文件的后缀名，例如 '.jpg'。
    :param file_suffix: YOLO 标签文件的后缀名，例如 '.txt'。
    :param save_draw_picture_path: 保存增强后图片的路径。
    :param expansion_pixel_value: 用于扩展边界框的像素值，边界框会按此像素值向外扩展。
    :param sampled_freq: 上下采样的次数，表示图像进行缩放的频率。
    :param target_size: 用于筛选小面积边界框的阈值，小于该阈值的框会被特殊处理。
    :return: None
    """
    for file_obj in os.listdir(file_path_txt):
        tmp_list = []
        file_name = os.path.splitext(file_obj)[0]
        width, height = img_size(os.path.join(file_path_jpg, file_name + picture_suffix))
        with open(os.path.join(file_path_txt, file_name + file_suffix), mode='r', encoding='utf-8') as file:
            splitlines = file.read().splitlines()
            for split_line in splitlines:
                split_list = split_line.split(' ')
                split_list = [float(x) for x in split_list]
                tmp_list.append(split_list)
        tmp_array = np.array(tmp_list)
        voc_array = yolo_x_center_y_center_x_y(tmp_array)
        image = cv2.imread(os.path.join(file_path_jpg, file_name + picture_suffix))
        voc_array[..., 1] = voc_array[..., 1] * width
        voc_array[..., 2] = voc_array[..., 2] * height
        voc_array[..., 3] = voc_array[..., 3] * width
        voc_array[..., 4] = voc_array[..., 4] * height
        voc_array = voc_array.astype(int)
        boxes, tmp_boxes, sampled_method_array = expansion_boxes(voc_array, expansion_pixel_value, width, height,
                                                                 target_size)
        boxes_loc_variable(image, boxes, tmp_boxes, sampled_method_array, sampled_freq, category_name,
                           distance, font_scale, file_name + picture_suffix, save_draw_picture_path)


def img_size(file_path: str):
    """
    读取图片的大小
    :param file_path:
    :return:
    """
    with Image.open(file_path) as img:
        return img.size


def expansion_boxes(voc_array, expansion_pixel_value: int, img_width: int, img_height: int, target_size: int):
    """
    对边界框进行外扩操作，并确保不会超出图像的边界。

    :param voc_array: 输入的边界框数组，格式为 [class_id, x_min, y_min, x_max, y_max]。
    :param expansion_pixel_value: 要扩展的像素值，即每个边界框四周需要扩展的像素数量。
    :param img_width: 图像的宽度，确保边界框不会超过图像的宽度。
    :param img_height: 图像的高度，确保边界框不会超过图像的高度。
    :param target_size: 目标的最小面积阈值，面积小于该值的边界框将被标记。
    :return: 返回外扩后的边界框数组 voc_array，坐标相对变化的边界框 voc_array_copy，以及面积标记数组 area_array。
    """
    area_array = (voc_array[..., 3] - voc_array[..., 1]) * (voc_array[..., 4] - voc_array[..., 2])
    area_array = np.where(area_array >= target_size, 0, 1).reshape(-1, 1)
    voc_array_copy = voc_array.copy()
    voc_array[..., 1] = np.maximum(voc_array[..., 1] - expansion_pixel_value, 0)
    voc_array[..., 2] = np.maximum(voc_array[..., 2] - expansion_pixel_value, 0)
    # 确保右下角坐标不超过图像边界
    voc_array[..., 3] = np.minimum(voc_array[..., 3] + expansion_pixel_value, img_width)
    voc_array[..., 4] = np.minimum(voc_array[..., 4] + expansion_pixel_value, img_height)
    voc_array_copy[..., 1:3] -= voc_array[..., 1:3]
    voc_array_copy[..., 3:5] -= voc_array[..., 1:3]
    return voc_array, voc_array_copy, area_array


def boxes_loc_variable(img, voc_array_pre, voc_last, sampled_method_array, sampled_freq: int, category_name: dict,
                       distance, font_scale, image_name, save_draw_picture_path):
    voc_concatenate = np.concatenate([voc_array_pre, voc_last, sampled_method_array], axis=1)
    k = 0

    for obj in voc_concatenate:
        screenshot_img = img[obj[2]:obj[4], obj[1]:obj[3]]
        screenshot_img, obj[..., 5:10] = up_sub_sampled(screenshot_img, obj[10], sampled_freq, obj[..., 5:10])
        save_yolo_file_and_image(screenshot_img, obj[..., 5:10], r"D:\pcb_data\data1\labels\labels_test",
                                 r'D:\pcb_data\data1\images\image_test',
                                 str(category_name.get(obj[0])) + "_" + str(k) + "_" + image_name, '.txt', '.jpg')
        draw_frame_on_picture(obj[..., 5:10].reshape(1, -1), screenshot_img,
                              category_name, distance, font_scale,
                              str(category_name.get(obj[0])) + "_" + str(k) + "_" + image_name,
                              save_draw_picture_path)
        k += 1


def up_sub_sampled(img, sampled_method: bool, sampled_freq: int, voc_array):
    """
    对图像进行上采样或下采样操作，并根据操作后的图像尺寸调整边界框坐标。
    :param img: 输入图像
    :param sampled_method: 采样方法的布尔值。如果为 True 则进行上采样（放大图像），为 False 则进行下采样（缩小图像）。
    :param sampled_freq: 采样频率，表示进行采样的次数。
    :param voc_array: 边界框数据，格式为 [class_id, x_min, y_min, x_max, y_max]，用于目标检测的标注。
    :return: 经过采样处理后的图像和调整过的边界框数组。
    """
    for i in range(sampled_freq):
        old_img = img.copy()
        if sampled_method:
            img = cv2.pyrUp(img)
            calculate_scaling_ratio(old_img, img, voc_array)
        else:
            img = cv2.pyrDown(img)
            calculate_scaling_ratio(old_img, img, voc_array)
    return img, voc_array


def save_yolo_file_and_image(img, voc_array, save_file_path: str, save_image_path: str, file_name: str,
                             file_suffix: str, image_suffix: str):
    """
    将图像及其对应的YOLO格式边界框数据保存到指定路径。
    :param img: 输入的图像（通常是经过处理后的图像，可能是经过尺寸调整或其他预处理）
    :param voc_array: 原始的VOC格式的边界框数组，格式为 [class_id, x_min, y_min, x_max, y_max]
    :param save_file_path: 保存YOLO格式标签文件的目录路径
    :param save_image_path: 保存处理后图像文件的目录路径
    :param file_name: 文件名，保存时使用的文件名（不包含后缀）
    :param file_suffix: 保存YOLO标签文件的后缀（通常为“.txt”）
    :param image_suffix: 保存图像文件的后缀（通常为“.jpg”或“.png”）
    :return: None
    """

    img, yolo_array = voc_to_yolo_array(img, voc_array)
    np.savetxt(os.path.join(save_file_path, file_name + file_suffix), yolo_array, fmt='%s')
    cv2.imwrite(os.path.join(save_image_path, file_name + image_suffix), img)


def calculate_scaling_ratio(old_img, img, voc_array):
    """
    根据图片的缩放比例，调整 VOC 格式的边界框坐标。

    :param old_img: 原始图像，shape 为 (height, width, channels)
    :param img: 缩放后的图像，shape 为 (height, width, channels)
    :param voc_array: VOC 格式的边界框数组，格式为 [class_id, x_min, y_min, x_max, y_max]
    其中 x_min, y_min 是左上角坐标，x_max, y_max 是右下角坐标
    :return: None，修改 voc_array 原地调整边界框的坐标
        """
    scale_x = img.shape[1] / old_img.shape[1]
    scale_y = img.shape[0] / old_img.shape[0]
    voc_array[..., 1] = (voc_array[..., 1] * scale_x).astype(int)
    voc_array[..., 3] = (voc_array[..., 3] * scale_x).astype(int)
    voc_array[..., 2] = (voc_array[..., 2] * scale_y).astype(int)
    voc_array[..., 4] = (voc_array[..., 4] * scale_y).astype(int)
