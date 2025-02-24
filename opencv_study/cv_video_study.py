import cv2

filename = r'C:\Users\Administrator\Downloads\WIN_20240731_20_35_31_Pro.mp4'
video_capture = cv2.VideoCapture(filename)
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if ret:
        frame = cv2.resize(frame, [521, 512])
        cv2.imshow('dd', frame)
        if cv2.waitKey(1) == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()
