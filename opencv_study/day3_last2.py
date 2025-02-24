"""
直方图均衡化, 把直方图的每个灰度级进行归一化处理，求每种灰度的累积分布，得到一个映射的灰度映射表，
然后根据相应的灰度值来修正原图中的每个像素.
"""

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def show(img, title=''):
    img = cv2.resize(img,[512,512])
    cv2.imshow(title, img)
    cv2.waitKey(0)
def get_equalizehist_img(imgname):
    """
    全局直方图
    :param imgname:
    :return:
    """
    img = cv2.imread(imgname,1)
    b,g,r = cv2.split(img)
    show(img,'old')
    b = cv2.equalizeHist(b) # 直方图均衡化
    g = cv2.equalizeHist(g)  # 直方图均衡化
    r = cv2.equalizeHist(r)  # 直方图均衡化
    dd = cv2.merge([b,g,r])
    show(dd,'equ')
    # # 直适应直方图均衡化 黑夜场景
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # cl1 = clahe.apply(img)
    # # cv2.imwrite('clahe_2.jpg', cl1)
    # show(cl1,'auto_equ')

"""
有效方法：3%+
"""
def rgbequalizehist_img(imgname):
    """区域直方图均衡化"""
    img = cv2.imread(imgname,1)
    b,g,r = cv2.split(img)
    show(img,'old')
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(1000, 1000))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    join = cv2.merge([b,g,r])
    show(join,'new')

rgbequalizehist_img("7f0c8d1aa4027903131c91e2e2dcecd541d19130123c10-72luVL_fw480webp.webp")