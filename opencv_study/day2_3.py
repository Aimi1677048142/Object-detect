"""
形态学变换：腐蚀、膨胀
# 只对灰度图，二值化后的图效果比较明显
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


img = cv2.imread('./qianming.png', 0)
show(img, 'o')
kernel = np.ones((3, 3), np.uint8)
# 腐蚀：腐蚀去毛刺
img1 = cv2.erode(img.copy(), kernel, iterations=2)
show(img1, '2')
# 膨胀：膨胀(变胖了)填小洞
img2 = cv2.dilate(img1.copy(), kernel, iterations=2)
show(img2, 'peng')

img3 = img2.copy() - img1.copy()  # 图像差值法: 胖 - 瘦 -》空心效果
show(img3, '3')
img4 = img.copy() - img2.copy()  # 原图-膨胀 -》毛刺
show(img4, '4')

# 开运算: 先腐蚀-》膨胀
opening = cv2.morphologyEx(img.copy(), cv2.MORPH_OPEN, kernel, iterations=2)
show(opening, 'open')
# 闭运算：先膨胀-》后腐蚀
closing = cv2.morphologyEx(img.copy(), cv2.MORPH_CLOSE, kernel, iterations=3)
show(closing, 'close')

# 形态学梯度：|膨胀-腐蚀|
gradient = cv2.morphologyEx(img.copy(), cv2.MORPH_GRADIENT, kernel, iterations=3)
show(gradient, 'gradient')

# 礼帽：|原始图像 - 开运算|
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

show(tophat, 'top')
# 黑帽：|原始图像 - 闭运算|
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
show(blackhat, 'black')

# kernel = np.ones((3, 3), np.uint8)
kernel = np.array([[0, 1, 1, 0, 1],
                   [0, 1, 0, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, 0, 1, 0],
                   [1, 0, 0, 0, 1]]).astype(np.uint8)
# 开运算。先腐蚀，再膨胀。 。
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
show(opening, 'openlast')
