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


blackMask = np.zeros([512, 512, 3], dtype=np.uint8)

lineImage = cv2.line(blackMask, (0, 0), (512, 512), (255, 0, 255), 1)
circleMask = cv2.circle(blackMask, (256, 256), 30, (0, 0, 255), -1)  # -1实心。半径小一点，画点 ***
rectangleImg = cv2.rectangle(blackMask, [10, 10, 20, 200], [0, 255, 0], 2)  # [leftx,lefty,w,h] ***
center = (100, 100)  # 中心点 (x, y)
axes = (50, 30)  # 长轴和短轴的长度
angle = 45  # 椭圆旋转角度
startAngle = 0  # 弧的起始角度
endAngle = 360  # 弧的结束角度
color = (0, 255, 0)  # 绿色
thickness = 2  # 边界的厚度

# 绘制椭圆
ellipseImg = cv2.ellipse(blackMask, center, axes, angle, startAngle, endAngle, color, thickness)
# todo1: 怎么绘一个中文
cv2.putText(blackMask, "hello中文", [110, 110], cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 255, 0], 2) # ***

# 绘制多边形  分割
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10], [100, 300]], np.int32)  # 坐标点
# 注意，pts要传入一个[pts]列表
img = cv2.polylines(blackMask, [pts], True, (255, 0, 0), 2)  # True和False是否起始到结束点 ***
show(img, 'polylines')
show(blackMask, '1')


#todo: 读一张图片A， 再图B的图片，B中取一个100x100的图像（MaskB)，将MaskB这个区域放到A图像中的任意起始位置
# 1. 完全粘贴(替换）
# 2. MaskB与A对应的位置addweights