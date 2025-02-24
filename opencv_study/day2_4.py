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


"""
低通滤波：中值、高斯、均值，得到模糊
高通滤波/图像梯度/滤波算子/纹理：转灰度图-》得取边缘、几何信息[形状]、轮廓信息
"""
img = cv2.imread('./010820005.jpg', 0)
img = cv2.resize(img, [512, 512])
ret,img_th = cv2.threshold(img.copy(), 40, 198, cv2.THRESH_BINARY)
show(img_th, 'th')
# 原图减去图像的均值
minValue = cv2.mean(img)

img = cv2.convertScaleAbs(img.copy() - minValue[0])

# show(img,'r')
show(img, 'X1')
img = cv2.medianBlur(img, 5)
x = cv2.Sobel(img, -1, 1, 0)  # dx, dy
# 转成正值
absX = cv2.convertScaleAbs(x)  # uint8
show(absX, 'X2')
y = cv2.Sobel(img, -1, 0, 1)
absY = cv2.convertScaleAbs(y)
show(absY, 'Y1')
#
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
show(dst, 'dst')
