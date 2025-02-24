# import cv2
# import numpy as np
# from PIL import Image
#
# from tool.base_tool import get_label_name, expansion_boxes, boxes_loc_variable
#
# file_path_txt = r'D:\tcl\pcb_data\data1\labels\val'
# file_path_picture = r'D:\tcl\pcb_data\data1\images\val'
# category = get_label_name(r'D:\tcl\pcb_data\data1\labels\classes.txt')
# if not category:
#     category = {x: f'category{x}' for x in range(10)}
# distance = 10
# font_scale = 0.5
# picture_suffix = ".jpg"
# file_suffix = ".txt"
# save_draw_picture_path = r'D:\pcb_data\data1\images\val_draw_frame1'
#
#
# def show(image, title, is_debug=True, is_scale=False):
#     if is_scale:
#         img_h, img_w = image.shape[0:2]
#         print(img_h, img_w)
#         image = cv2.resize(image, [int(img_h * 0.5), int(img_w * 0.5)])
#     if is_debug:
#         cv2.imshow(title, image)
#         cv2.waitKey(0)
#
#
# def yolo2coordinates(x, y, w1, h1, img_w, img_h):
#     xmin = round(img_w * (x - w1 / 2.0))
#     xmax = round(img_w * (x + w1 / 2.0))
#     ymin = round(img_h * (y - h1 / 2.0))
#     ymax = round(img_h * (y + h1 / 2.0))
#     # print(xmin, ymin, xmax, ymax)
#     return xmin, ymin, xmax, ymax
#
#
# def img_size(file_path: str):
#     with Image.open(file_path) as img:
#         return img.size
#
#
# filename = r'D:\pcb_data\data1\labels\val\1fcdab66-1103-4585-913e-1711c09f9234.txt'
# image_path = r'D:\pcb_data\data1\images\val\1fcdab66-1103-4585-913e-1711c09f9234.jpg'
# yolo_list = []
# with open(filename, mode='r', encoding='utf-8') as file:
#     for obj in file.read().splitlines():
#         yolo_list = [float(x) for x in obj.split(' ')]
# image_w, image_h = img_size(image_path)
# x1, y1, x2, y2 = yolo2coordinates(yolo_list[1], yolo_list[2], yolo_list[3], yolo_list[4], image_w, image_h)
# voc_array = np.array([[0, x1, y1, x2, y2]])
# boxes, tmp_boxes = expansion_boxes(voc_array, 50)
# image = cv2.imread(image_path)
# boxes_loc_variable(image, boxes, tmp_boxes, True, 2, category,
#                        distance, font_scale, '1fcdab66-1103-4585-913e-1711c09f9234', save_draw_picture_path)
# # cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
# # # 在图像上显示标签名
# # cv2.putText(image, str(yolo_list[0]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# #
# # show(image, 'box1')
