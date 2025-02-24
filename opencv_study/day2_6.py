"""
面试小问题：
canny:
   1. 高斯去噪
   2. sobel算子，多了一方角度
   3. 相同方向进行非极大值抑制
   4. 双域值管理，高于最大值保留，小于最小值去掉，中间值看是否连续
"""
import cv2
import numpy as np
def show(img, title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)


# 做形态学的前提条件，是一个二值化。
img = cv2.imread('./010820005.jpg', 0)
img = cv2.resize(img,[512,512])
img = cv2.medianBlur(img, 5)
dst = cv2.Canny(img.copy(), 50, 200)
show(dst, 'canny1')
dst = cv2.Canny(img.copy(), 150, 230)
show(dst, 'canny2')
