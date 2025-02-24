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

# 二值化：将图像变成指定的多个（至少）灰度值
# 让图像特征变得更明显

# 特征：像素值
# 一般只针对灰度图
img = cv2.imread("1.jpg", 0)
# 自定义代码操作二值化。 灵活性很可

def splitValue3(img):
    h, w = img.shape[:2]
    for j in range(h):
        for i in range(w):
            pixels = img[j, i]
            if pixels < 50:
                img[j, i] = 0
            elif pixels > 200:
                img[j, i] = 255
            else:
                # 中间的可以不变
                img[j, i] = img[j, i] * 0.8
                pass
    return img

def splitValue2(img):
    h, w = img.shape[:2]
    for j in range(h):
        for i in range(w):
            pixels = img[j, i]
            if pixels < 40:
                img[j, i] = 0
            else:
                img[j, i] = 255
    return img
# img2 = splitValue2(img)
# python推荐以下
# img[img<45] = 0
# img[img>45] = 255
# show(img,'1')


# 利用opencv封装好的2值化
# 标识成功
# cv2.THRESH_BINARY,超过40的取255。否则为0
# cv2.THRESH_BINARY_INV 超过40的取0，否则为255
#
# cv2.THRESH_TRUNC,超过40的为255，否则为40
# cv2.THRESH_TOZERO， 小于阈值的为0，否则不变
# cv2.THRESH_TOZERO_INV, 大于阈值的为0,否则不变
ret, thImg1 = cv2.threshold(img.copy(), 40, 198, cv2.THRESH_BINARY)
show(thImg1,'1')

ret, thImg2 = cv2.threshold(img.copy(),40, 255, cv2.THRESH_BINARY_INV)
show(thImg2,'2')

ret, thImg3 = cv2.threshold(img.copy(),40,255,cv2.THRESH_TRUNC)
show(thImg3,'3')

ret, thImg4 = cv2.threshold(img.copy(),40,255,cv2.THRESH_TOZERO)
show(thImg4,'4')

ret, thImg5 = cv2.threshold(img.copy(),40,255,cv2.THRESH_TOZERO_INV)
show(thImg5,'5')






