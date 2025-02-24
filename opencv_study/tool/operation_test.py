import os

import cv2
import numpy as np

from opencv_operation.operation import letter_box, get_image_category, yolo_box_to_original_image, slide_crop, \
    copy_boxes_to_image, image_mosic, split_image_list, cv_equalize_rgb_histogram, slip_cutting, cv_apply_clahe_to_rgb, \
    random_brightness_adjust, image_mean, copy_boxes_to_image_by_mask, laplace_blur, gaussian_blur, mean_blur, \
    edge_enhancement, geometric_transformation_flip, random_translate_image
from tool.base_tool import get_label_name, img_size, yolo_x_center_y_center_x_y, draw_frame_on_picture

file_path_txt = r'D:\pcb_data\original_ img and yolo_imfo\yolo_label_0902'
file_path_jpg = r'D:\pcb_data\original_ img and yolo_imfo\original img'
classes_path = r'D:\pcb_data\original_ img and yolo_imfo\yolo_label_0902\classes.txt'
category = get_label_name(classes_path)
if not category:
    category = {x: f'category{x}' for x in range(10)}
distance = 10
font_scale = 0.5
picture_suffix = ".jpg"
file_suffix = ".txt"
save_draw_picture_path = r'D:\pcb_data\original_ img and yolo_imfo\test11'
result_list = []
for file_obj in os.listdir(file_path_txt):

    tmp_list = []
    file_name = os.path.splitext(file_obj)[0]
    if "classes" == file_name:
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
    # image1, boxes = cv_equalize_rgb_histogram(image, voc_array.astype(int))
    # image1, boxes = cv_apply_clahe_to_rgb(image, 2.0, (8, 8), voc_array.astype(int))
    # image1, boxes = random_brightness_adjust(image, 0.5, 1, voc_array.astype(int))
    # image1, boxes = image_mean(image, voc_array.astype(int))
    # image1, boxes = laplace_blur(image, voc_array.astype(int), 3, 0, 0.5, 0.5)
    # image1, boxes = gaussian_blur(image, voc_array.astype(int), 9, 0.5, 0.2, 0)
    # image1, boxes = mean_blur(image, voc_array.astype(int), 3, 0.5, 0.2,0)
    # image1, boxes = edge_enhancement(image, voc_array.astype(int), 3, 0.5, 0.2)
    # image1, boxes = geometric_transformation_flip(image, voc_array.astype(int), 0)
    # image1, boxes = random_translate_image(image, voc_array.astype(int), [0,100], [0,100])
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 检验letterbox
    new_image, scale_ratio, boxes, top, bottom, left, right = letter_box(image, [640, 640], voc_array)
    # 检验letterbox
    # draw_frame_on_picture(boxes.astype(int), image1,
    #                       get_image_category(classes_path),
    #                       distance, font_scale,
    #                       file_name + picture_suffix,save_draw_picture_path)
    # 检验还原letterbox
    # yolo_box_to_original_image(image, boxes, scale_ratio, top, left,
    #                            classes_path, distance,
    #                            font_scale, file_name + picture_suffix, save_draw_picture_path)
    # 检验滑动裁图
    # cropped_images, update_boxes = slide_crop(image, voc_array, 200, [1280, 1280])
    # for i in range(len(update_boxes)):
    #     draw_frame_on_picture(update_boxes[i].astype(int),cropped_images[i], get_image_category(classes_path), distance, font_scale,
    #                           file_name+f"caijian{i}" + picture_suffix, save_draw_picture_path)

    # cutting_image, cutting_boxes = slip_cutting(image, voc_array.astype(int), 200, [1280, 1280], 2)
    # for i in range(len(cutting_boxes)):
    #     draw_frame_on_picture(cutting_boxes[i].astype(int), cutting_image[i], get_image_category(classes_path),
    #                           distance, font_scale, file_name + f"caijian{i}" + picture_suffix, save_draw_picture_path)
    # 检验复制n份boxes
    # for i in range(len(voc_array)):
    # new_image, result_box = copy_boxes_to_image(image, voc_array[0].astype(int), 3, 0.5)
    # new_image, result_box = copy_boxes_to_image_by_mask(image, voc_array[0].astype(int), 3)
    # draw_frame_on_picture(result_box.astype(int), new_image,
    #                       get_image_category(
    #                           classes_path),
    #                       distance, font_scale,
    #                       file_name + picture_suffix, save_draw_picture_path)
    # 检验mosaic
    # tuple1 = tuple([image, voc_array.astype(int)])
    # result_list.append(tuple1)

# mosaic检验
# i = 0
# for obj in split_image_list(result_list, group_size=4):
#     mosic_image, boxes_list = image_mosic(obj[0][0], obj[0][1], obj[1][0], obj[1][1], obj[2][0], obj[2][1], obj[3][0],
#                                           obj[3][1])
#
#     draw_frame_on_picture(boxes_list.astype(int), mosic_image, get_image_category(classes_path), distance,
#                           font_scale,
#                           f"new{i}.jpg", save_draw_picture_path)
#     i += 1
