"""
特征：由像素构成的值，一组像素值构成种几何形状
1. 像素值
    1. 平均灰度值
    2. 颜色值
    3. 语义特征（高级特征）
    
2. 几何特征
   1. 面积
   2. 线条的周长
   3. 长宽比
   4. 轮廓面积占外接矩形的面积
   5. 方向
   6. 端点
   7. 凸包
   8. 外接/内接矩形
   9. 外接/内接圆
   10. 外接/内接椭圆
   11. 低级特征
   
ANN决定类别，由高级和低级特征共同决定; opencv主要靠低级特征（尽量获取几何信息来做判断）;
"""

import numpy as np
import cv2


def show(img, title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)


# 边缘检测都是针对灰度图像
img = cv2.imread("7f0c8d1aa4027903131c91e2e2dcecd541d19130123c10-72luVL_fw480webp.webp", 0)
# img = cv2.resize(img, [512, 512])
show(img, '1')
oldImg = img.copy()  # 创建1个新对象
gau_shi = cv2.GaussianBlur(oldImg, [3, 3], 0)
_, gau_shi = cv2.threshold(gau_shi, 128, 255, cv2.THRESH_BINARY)
canny = cv2.Canny(gau_shi, 50, 150)
show(canny, 'canny')
contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    if len(contours[i]) < 100:
        continue
    # oldImg.copy()  # ，每次运行的时候，会新建一个地址空间。避免内容叠加

    # img = cv2.drawContours(oldImg.copy(), contours, i, (0, 0, 255), 2)
    # show(img, str(i))
    # 外接矩形
    leftx, lefty, w, h = cv2.boundingRect(contours[i])  # 得到最小外接矩型 ,不包含旋转矩型
    img = cv2.rectangle(img, (leftx, lefty), (leftx + w, lefty + h), (0, 0, 255), 20)
    show(img, "rectangle")

    # 框可以有角度，旋转框
    # xy,wh,angle = cv2.minAreaRect(contours[i]) # 返回回点集cnt的最小外接矩形
    # print(angle)
    # box = cv2.boxPoints([xy,wh,angle])  # 获取最小外接矩形的4个顶点坐标
    # box = box.astype(int)
    # img2 = cv2.polylines(oldImg.copy(),[box],True,(255,255,0),5)
    # show(img2,'xuanzhuan')

    # (cx, cy), radius = cv2.minEnclosingCircle(contours[i]) # 得到最小外接圆
    # print(radius)
    # center = (int(cx), int(cy))
    # radius = int(radius)
    # img = cv2.circle(img, center, radius, (0, 255, 255), 20)
    # show(img,'yuan')
    # """
    # （x, y）代表椭圆中心点的位置
    # （a, b）代表长短轴长度，应注意a、b为长短轴的直径，而非半径
    #  angle 代表了中心旋转的角度
    # """
    # ellipse = cv2.fitEllipse(contours[i]) # 得到椭圆
    # im = cv2.ellipse(oldImg.copy(), ellipse, (0, 255, 0), 2)
    # show(im,'tuoyuan')

    # 根据一组点拟合出一条直线
    # rows, cols = img.shape[:2]
    # [vx, vy, x, y] = cv2.fitLine(contours[i], cv2.DIST_L2, 0, 0.01, 0.01)
    # # 提取数组中的标量值
    # vx = vx.item()
    # vy = vy.item()
    # x = x.item()
    # y = y.item()
    # lefty = int((-x * vy / vx) + y)
    # righty = int(((cols - x) * vy / vx) + y)
    # img = cv2.line(oldImg.copy(), (cols - 1, righty), (0, lefty), (255, 255, 0), 2)
    # show(img,'ff')
