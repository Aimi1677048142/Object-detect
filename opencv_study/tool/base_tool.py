import logging
import os
import random

import numpy as np
import cv2
from PIL import Image

# 配置日志记录器
logging.basicConfig(level=logging.DEBUG)


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


def yolo2coordinates(x, y, w1, h1, img_w, img_h):
    """
    yolo转换为voc
    :param x:
    :param y:
    :param w1:
    :param h1:
    :param img_w:
    :param img_h:
    :return:
    """
    xmin = round(img_w * (x - w1 / 2.0))
    xmax = round(img_w * (x + w1 / 2.0))
    ymin = round(img_h * (y - h1 / 2.0))
    ymax = round(img_h * (y + h1 / 2.0))
    return xmin, ymin, xmax, ymax


def img_size(file_path: str):
    """
    读取图片的大小
    :param file_path:
    :return:
    """
    with Image.open(file_path) as img:
        return img.size


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


def draw_frame_on_picture(array_list, img,
                          category_name: dict,
                          distance: int,
                          font_scale: float,
                          image_name: str,
                          save_draw_picture_path=None):
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


def draw_frame_to_picture(file_path_txt: str, file_path_jpg: str, category_name: dict,
                          distance: int,
                          font_scale: float,
                          picture_suffix: str,
                          file_suffix: str,
                          save_draw_picture_path: str):
    """
    给图片进行画框
    :param file_path_txt:
    :param file_path_jpg:
    :param category_name:
    :param distance:
    :param font_scale:
    :param picture_suffix:
    :param file_suffix:
    :param save_draw_picture_path:
    :return:
    """
    for file_obj in os.listdir(file_path_txt):
        tmp_list = []
        file_name = os.path.splitext(file_obj)[0]
        if file_name == "classes":
            continue
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
        draw_frame_on_picture(voc_array.astype(int), image, category_name, distance, font_scale,
                              file_name + picture_suffix, save_draw_picture_path)


def get_label_name(file_path: str) -> dict:
    """
    获取label的名称并且拼装成字典
    :param file_path:
    :return:
    """
    label_dict = {}
    if not os.path.exists(file_path):
        return label_dict
    with open(file_path, mode='r', encoding='utf-8') as file:
        label_dict = {index: value for index, value in enumerate(file.read().splitlines())}
    return label_dict


def split_binary(img, pixel_value1: int, pixel_value2: int, pixel_value_to_judge: int):
    """
    图像的数据处理，二值化
    :param img:
    :param pixel_value1:
    :param pixel_value2:
    :param pixel_value_to_judge:
    :return:
    """
    img_h, img_w = img.shape[:2]
    for i in range(img_h):
        for j in range(img_w):
            img[i, j] = pixel_value1 if img[i, j] > pixel_value_to_judge else pixel_value2
    return img


def replace_picture(image1, image2, size, flag, alpha: float, beta: float):
    """
    随机替换图片
    :param image1:
    :param image2:
    :param size:
    :param flag:
    :param alpha:
    :param beta:
    :return:
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


def expansion_boxes(voc_array, expansion_pixel_value: int, img_width: int, img_height: int, target_size: int):
    """
    外扩像素值
    :param voc_array:
    :param expansion_pixel_value:
    :param img_width
    :param img_height
    :param target_size
    :return:
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


def up_sub_sampled(img, sampled_method: bool, sampled_freq: int, voc_array):
    for i in range(sampled_freq):
        old_img = img.copy()
        if sampled_method:
            img = cv2.pyrUp(img)
            calculate_scaling_ratio(old_img, img, voc_array)
        else:
            img = cv2.pyrDown(img)
            calculate_scaling_ratio(old_img, img, voc_array)
    return img, voc_array


def calculate_scaling_ratio(old_img, img, voc_array):
    scale_x = img.shape[1] / old_img.shape[1]
    scale_y = img.shape[0] / old_img.shape[0]
    voc_array[..., 1] = (voc_array[..., 1] * scale_x).astype(int)
    voc_array[..., 3] = (voc_array[..., 3] * scale_x).astype(int)
    voc_array[..., 2] = (voc_array[..., 2] * scale_y).astype(int)
    voc_array[..., 4] = (voc_array[..., 4] * scale_y).astype(int)


def boxes_loc_variable(img, voc_array_pre, voc_last, sampled_method_array, sampled_freq: int, category_name: dict,
                       distance, font_scale, image_name, save_draw_picture_path):
    voc_concatenate = np.concatenate([voc_array_pre, voc_last, sampled_method_array], axis=1)
    k = 0
    yolo_result = []
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


def save_yolo_file_and_image(img, voc_array, save_file_path: str, save_image_path: str, file_name: str,
                             file_suffix: str, image_suffix: str):
    """
    保存文件
    :param img:
    :param voc_array:
    :param save_file_path:
    :param save_image_path:
    :param file_name:
    :param file_suffix:
    :param image_suffix:
    :return:
    """

    img, yolo_array = voc_to_yolo_array(img, voc_array)
    np.savetxt(os.path.join(save_file_path, file_name + file_suffix), yolo_array, fmt='%s')
    cv2.imwrite(os.path.join(save_image_path, file_name + image_suffix), img)


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
     多尺度图像增加
    :param file_path_txt:
    :param file_path_jpg:
    :param category_name:
    :param distance:
    :param font_scale:
    :param picture_suffix:
    :param file_suffix:
    :param save_draw_picture_path:
    :param expansion_pixel_value:
    :param target_size:
    :param sampled_freq:
    :return:
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

