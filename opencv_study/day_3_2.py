import numpy as np
import cv2


# 轮廓检测
def show(img, title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)


img = cv2.imread('./yuan.png', 1)
img = cv2.resize(img, [512, 512])
orgImg = img.copy()  # 创建一个深copy对象

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0)
# show(img,'gauss')
_, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
show(img, 'thres')
# canny = cv2.Canny(img, 50, 150) # 可选
# show(canny, 'Canny')
# show(img, 'canny')
# cv2.RETR_EXTERNAL 只检测外轮廓
# cv2.RETR_LIST  没有层级关系的轮廓
# cv2.RETR_CCOMP 有层级关系的轮廓
# cv2.RETR_TREE  树结构的轮廓

# cv2.CHAIN_APPROX_NONE	存储所有边界点
# cv2.CHAIN_APPROX_SIMPLE	压缩垂直、水平、对角方向，只保留端点

# contours:轮廓点。列表格式，每一个元素为一个3维数组（其形状为（n,1,2），其中n表示轮廓点个数，2表示像素点坐标）,表示一个轮廓
# hierarchy:轮廓间的层次关系,为三维数组，形状为（1,n,4），其中n表示轮廓总个数，4指的是用4个数表示各轮廓间的相互关系
# 第一个数表示同级轮廓的下一个轮廓编号，第二个数表示同级轮廓的上一个轮廓的编号，
# 第三个数表示该轮廓下一级轮廓的编号，第四个数表示该轮廓的上一级轮廓的编号。
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 一定要先二值化
# 遍历轮廓，还原到原图中，人眼识别在那里
for i in range(len(contours)):
    # orgImg = cv2.drawContours(orgImg, contours[i], -1, (0, 0, 255), 3)
    # show(orgImg, str(i))
    # # 1.特征，计算轮廓的面积【*****】
    area = cv2.contourArea(contours[i])
    print(area)
    if area > 5000 and area < 8000:

        print(f"轮廓的面积{i}：", area)
        # rgb图,  所有轮廓, 第几个轮廓
        # 为了调试，过滤才使用
        perimeter = cv2.arcLength(contours[i], True)
        print(f'当前轮廓的周长:{perimeter}')
        if perimeter > 1000: continue
        orgImg = cv2.drawContours(orgImg, contours, i, (0, 0, 255), 3)

show(orgImg, 'dd')

pass
