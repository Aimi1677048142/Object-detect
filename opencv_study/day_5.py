import numpy as np
import cv2

def show(img, title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)

# 实现：实时红色物体跟踪
cap = cv2.VideoCapture(0)
# https://tool.yovisun.com/rgbcolor/
while True:
    # 读图像数据,ret状态，是否成功
    ret, frame = cap.read()
    if ret:
        # 转换为灰度图
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # show(hsv)
        # 红色阈值范围
        lower_blue = np.array([156, 43, 46])
        upper_blue = np.array([180, 255, 255])

        mask = cv2.inRange(hsv,lower_blue,upper_blue)
        # 对原图像和掩模进行位运算
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # 显示
        cv2.imshow('frame', res)

    # 0xFF 16进制，就是q
    # ord()获取某个字符的16进制
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    pass
    """
    1. 外接矩形，轮廓的面积，内接矩形，外接圆
    2. 宽高比，周长和面积比，夹角
    """