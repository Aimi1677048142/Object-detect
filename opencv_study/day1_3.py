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


# 颜色空间转换
img = cv2.imread("1.jpg")  # bgr
# imgRGB = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB) # rgb
# # img = np.hstack([img.copy(),imgRGB])
# #
imgHSV = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
# 色相，饱和度，明度 面试问到过
# Hue：0-179,其它不变
imgHSL = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HLS)
# imgGray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
# imgHSL = np.hstack([imgGray])

# 将imgHSV和imgHSL按一定比例加在一起

imgNew = cv2.addWeighted(img.copy(), 0.4, imgHSV, 0.6, 0)
imgNew = cv2.cvtColor(imgNew, cv2.COLOR_BGR2RGB)

# 　不同的图片也可以放在一起
img2 = cv2.imread(r"E:\data_source\3D_print_data\UC02\0821\jieou\2.jpg")

img2 = cv2.resize(img2, imgNew.shape[:2][::-1])

img3 = cv2.addWeighted(imgNew.copy(), 0.4, img2.copy(), 0.6, 0)  # mixup 是一种有效的数据增加方法

# 　将一个图像变暗，或者变亮

img2_1 = (img2.copy() * 0.5).astype("uint8")
imgNew_1 = (img.copy() * 1.2).astype("uint8")

imgNew_2 = (img2_1 * 0.3 + imgNew_1 * 0.5).astype("uint8")
show(imgNew_2, 'd')
