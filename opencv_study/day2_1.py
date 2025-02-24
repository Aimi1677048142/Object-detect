"""
 卷积计算
 1. 卷积核的大小; 卷积里面的值（参数 opencv固定的）;CNN里面w学习到的;
 2. 卷积核越小，感觉的区域越小，提取细粒特征，特征保留情况会更好； 卷积核越大，提取的颗粒度特征(粗）
 3. 在opencv卷积之后，图像有可能变得更平滑（相邻像素之间的梯度很小）
    也有可能更尖锐（相领像素之间的梯度差异较大），关键在于卷积核里面的参数设置

"""

import cv2
import numpy as np


def show(img, title='', isDebug=True, isScale=False):
    if isScale:
        h, w = img.shape[0:2]
        print(h, w)
        # 宽和高
        img = cv2.resize(img, [int(w * 0.5), int(h * 0.5)])
    # bgr
    if isDebug:
        cv2.imshow(title, img)  # 展示
        cv2.waitKey(0)  # 等待一个按键


img = cv2.imread(r"D:\image\lenaNoise.png")
img = cv2.resize(img, [512, 512])

# kernel = np.ones([3, 3], np.float32)/30

kernel = np.array(
    [[0, 0.2, 0],
     [0, 0.12, 10],
     [0, 0.002, 0],
     ]
)
img2 = cv2.filter2D(img, -1, kernel)

# show(img2, '3')
for _ in range(3):
    #
    kernel = np.ones([3, 10], np.float32) / 30
    img = cv2.filter2D(img, -1, kernel)
show(img)
