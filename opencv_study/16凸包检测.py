import numpy as np
import cv2
# 轮廓检测
def show(img, title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)


# 边缘检测都是针对灰度图像
img = cv2.imread("4787.jpg", 0)
show(img,'1')
oldImg = img.copy()  # 创建1个新对象
gau_shi = cv2.GaussianBlur(oldImg, [3, 3], 0)
canny = cv2.Canny(gau_shi, 50, 150)
show(canny)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    # if cv2.isContourConvex(contours[i]): # 这个轮廓是不是凸的
        hull    = cv2.convexHull(contours[i]) # 得到凸包
        for h in hull: #遍历每个凸包，绘点
            oldImg = cv2.circle(oldImg, h[0], 2, [0, 0, 255], 5)
# show(oldImg,'all')

# for i in range(len(contours)):
#     # oldImg.copy()，每次运行的时候，会新建一个地址空间。避免内容叠加
#     img = cv2.drawContours(oldImg.copy(), contours, i, (0,0,255), 3)
#     c.show(img,i)
best_contours = contours[12]
img = cv2.drawContours(oldImg.copy(), contours, 12, (0, 0, 255), 3)
show(img,'draw')
hull = cv2.convexHull(best_contours)  # 得到凸包，顶点
for h in hull:
    oldImg = cv2.circle(oldImg, h[0], 5, [0, 0, 255], 5)
show(oldImg, 'point')
