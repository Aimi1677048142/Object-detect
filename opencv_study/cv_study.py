import cv2
import numpy as np


def show(image, title, is_debug=True, is_scale=False):
    if is_scale:
        img_h, img_w = image.shape[0:2]
        print(img_h, img_w)
        image = cv2.resize(image, [int(img_h * 0.5), int(img_w * 0.5)])
    if is_debug:
        cv2.imshow(title, image)
        cv2.waitKey(0)


img = cv2.imread(r'D:\image\beauty-8870258_1280.jpg')
b, g, r = cv2.split(img)
# show(b, 'b')
# show(g, 'g')
# show(r, 'r')
# show(img, 'img')
image_merge = cv2.merge([r, g, b])
# show(image_merge,'image_merge')
image_dstack = np.dstack([r, g, b])
# show(image_dstack,'image_dstack')
img2 = img.copy()
# show(img2[...,0],'b')
# show(img2[...,1],'g')
# show(img2[...,2],'r')
img2[..., 0] = 0
# show(img2,'image2')
img3 = img2[10:50, 50:100, :]
# show(img3,'img2')
img2[10:50, 50:100] = 144
show(img2, '144')
cv2.imwrite('new.jpg', img2)
