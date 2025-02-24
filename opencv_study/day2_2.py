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


# 去噪声 -> 模糊
# 均值滤波（均值模糊）

img = cv2.imread("lenaNoise.png")
img = cv2.resize(img, [512, 512])
# 均值滤波
result = cv2.blur(img.copy(), ksize=(5, 5))  # 求5x5区域之间的平均值. 卷积核的参数就是全1
show(result, 'junzhi')

# 方框滤波。如果将图像归一化，效果与均值滤波一样；如果不归一化，超过255的用255来表示.
# 对于一个大小为 3x3 的方框滤波器，所有的 9 个元素的值都是 1/9
img2 = cv2.boxFilter(img.copy(), -1, ksize=[5, 5], normalize=True)  # 255->1/255 -> 1 * 1/25 -> 1/25 * 255
img3 = cv2.boxFilter(img.copy(), -1, ksize=[5, 5], normalize=False)

show(img2, 'f1')
show(img3, 'f2')

# 高斯滤波（高期模糊）

img4 = cv2.GaussianBlur(img.copy(), [5, 5], 1, 0)
img4 = cv2.GaussianBlur(img4.copy(), [5, 5], 1, 0)
img4 = cv2.GaussianBlur(img4.copy(), [5, 5], 1, 0)
show(img4, 'gaoshi')

# 排序
# 椒盐噪声效果不错
img5 = cv2.medianBlur(img.copy(), 3)  # 按3x3，卷积核的值是1，但是取中间值。没有相加这个结果
show(img5, 'z')
# 双边滤波
blur = cv2.bilateralFilter(img.copy(), 9, 75, 75)
show(blur, 'b')
nn = cv2.fastNlMeansDenoisingColored(img.copy())
show(nn, 'nn')
