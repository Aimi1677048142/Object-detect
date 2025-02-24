import cv2

import numpy as np

# 视频：连续的图像+声音(opencv无法处理声音,ffmpeg音视频）
# 帧率(fps)：1秒时间内能够处理连续图片数  30fps
# 人眼看一个东西，实时， 15fps

# 0当前摄像头
# filename本地的视频
# rtsp流媒体：
# 推流：ipc(智能摄像头)->网络(上传)->服务器中心(url)
# 拉流：app->服务器中心拿到一个地址url(下载)
filename = r"E:\data_source\3D_print_data\UC02\0731\WIN_20240731_20_35_31_Pro.mp4"
url = "http://devimages.apple.com/iphone/samples/bipbop/gear1/prog_index.m3u8" # rtsp
vc = cv2.VideoCapture(url)

while vc.isOpened():
    # ret表示是否读成功
    # frame就是图像
    ret, frame = vc.read()
    if ret:
        frame = cv2.resize(frame, [512, 512])
        cv2.imshow("shan_ge", frame)
        if cv2.waitKey(1)==ord("q"):
            break
vc.release() # 释放视频流
cv2.destroyAllWindows() # 关闭窗口