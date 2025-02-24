"""
0. 传统图像 + 机器学习
   opencv[开源]\haclon[工业缺陷检测]+c#
   opencv: 图像的前处理、后处理，数据增强，工业上的检测[泛化能力较差]
   先了解主要函数的功能，以及展示出来的效果，不要太专注于原理；
   做几个小demo

图像/视频领域：
1. 分类  5
   给整张图片分类

2. 识别（目标识别、目标检测、缺陷识别、物体识别....） 12
   从一张图片中，找到目标所在的位置（矩形框）

3.  分割(语义/实例）

4.  关键点识别/姿态识别[行为识别：目标检测+姿态识别]  7

5.  图片生成(AIGC: 生成式网络，图片生成，文本生成，视频生成，语音生成） -》大模型  15

6.  多模态   20
    将图片、声音、文本、视频，至少2者结合; 2D和3D; C++

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


# opencv 默认目录文件不支持中文
# 读图像 uint8: 0 - 255之间
img1 = cv2.imread(r"E:\data_source\3D_print_data\UC02\0821\jieou\2.jpg", cv2.IMREAD_COLOR)  # 按bgr cv2.IMREAD_COLOR
b, g, r = cv2.split(img1)  # 分割b, g, r 从不同的光照去看。BGR在一起看不清楚，可采用分隔的方式
# show(b, 'b')
# show(g, 'g')
# show(r, 'r')
joinImg = cv2.merge([r, g, b])  # yolo它的格式是rgb
# show(joinImg, 'j')
joinImg2 = np.dstack([b, g, r])
# show(joinImg2, 'j2')

img2 = img1.copy()  # 复制一份新的。深copy
show(img2[..., 0], 'b')  # b
show(img2[..., 1], 'g')
show(img2[..., 2], 'r')
img2[..., 0] = 0  # 修改某个图

# 截图 y1:y2, x1:x2
img3 = img2[10:50, 3:60, :]

show(img3, 'all')
img2[10:50, 3:60, :] = 0  # 对某个区域进行填0操作
show(img2, 'mask')

# show
# h, w
img2 = cv2.imread("1.jpg", 0)  # 灰度图，没有颜色 cv2.IMREAD_GRAYSCALE
# show(img2,"1")
img2[10:50, 3:60] = 144
cv2.imwrite("new.jpg", img2)
show(img2, '144')
pass
