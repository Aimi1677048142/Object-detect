import cv2
import numpy as np


def show(image, title, is_debug=True, is_scale=False):
    if is_scale:
        img_h, img_w = image.shape[0:2]
        print(img_h, img_w)
        image = cv2.resize(image, [int(img_h * 0.5), int(img_w * 0.5)])
    if is_debug:
        cv2.imshow(title, image)
        cv2.waitKey(0)


def split_multiple_values(img):
    image = cv2.imread(img)
    img_h, img_w = cv2.imread(img).shape[:2]
    for i in range(img_h):
        for j in range(img_w):
            if image[i, j] < 40:
                image[i, j] = 0
            elif image[i, j] > 200:
                image[i, j] = 255
            else:
                image[i, j] = image[i, j] * 0.8
    return image


def split_binary(img):
    image = cv2.imread(img, 0)
    img_h, img_w = image.shape[:2]
    for i in range(img_h):
        for j in range(img_w):
            pixels = image[i, j]
            if pixels < 40:
                image[i, j] = 0
            else:
                image[i, j] = 255
    return image


filename = r'D:\image\beauty-8870258_1280.webp'
image1 = split_binary(filename)
image2 = cv2.imread(filename, 0)
image2[image2 < 40] = 0
# 小于40为0，否则为255
ret, th_image1 = cv2.threshold(image1.copy(), 40, 255, cv2.THRESH_BINARY)
# 超过40为255，否则为40
ret1, th_image2 = cv2.threshold(image1.copy(), 40, 255, cv2.THRESH_BINARY_INV)
# 小于40不变，超多40为0
ret3, th_image3 = cv2.threshold(image1.copy(), 40, 255, cv2.THRESH_TRUNC)
# 超过40不变，否则为0
ret4, th_image4 = cv2.threshold(image1.copy(), 40, 255, cv2.THRESH_TOZERO)
# 超多40为0，否则不变
ret5, th_image5 = cv2.threshold(image1.copy(), 40, 255, cv2.THRESH_TOZERO_INV)
show(th_image5, '1')
