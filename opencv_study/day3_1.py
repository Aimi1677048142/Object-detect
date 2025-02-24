import numpy as np
import cv2

"""
# todo:
多尺度图像增加: 将一个图等比例变大，或者等比例变小 上涨3%+ 10%+
1. 将原图比较大的大图，下采样1次或者2次, boxes也会发生变化
2. 将原图比较小的图，上采样1次或者2次, boxes也会发生变化
3. 切目标区域boxes的内容切下来，外扩100个像素(函数参数)，上采样1倍或者2倍，boxes也会发生变化[近景、大目标]

少1/2。多2/1
"""
import cv2
import numpy as np
def show(img, title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)


img = cv2.imread("./010820005.jpg")
show(img,'old')
for _ in range(2):
    img = cv2.pyrDown(img) # w,h等比例缩小一倍
show(img,'0')

img2 = cv2.imread('./20240829101428.jpg')
for _ in range(2):
    img2 = cv2.pyrUp(img2)
show(img2,'n')