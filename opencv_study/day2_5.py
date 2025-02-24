import cv2
import numpy as np


def show(img, title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)


# 做形态学的前提条件，是一个二值化。
img = cv2.imread('./lenaNoise.png', 0)
img = cv2.medianBlur(img, 5)
img = cv2.resize(img, [512, 512])
dst = cv2.Laplacian(img, -1, ksize=3)
dst = cv2.convertScaleAbs(dst)
show(dst, 'Laplacian')
