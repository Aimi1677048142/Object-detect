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


image1 = cv2.imread(r'D:\image\beauty-8870258_1280.webp')
image_rgb = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2RGB)
image_hstack = np.hstack([image1.copy(), image_rgb])
image_hsv = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2HSV)
image_hls = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2HLS)
image_gray = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2GRAY)
image_weighted = cv2.addWeighted(image_hsv.copy(), 0.4, image_hls.copy(), 0.6, 0)
image_weighted = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2RGB)
image2 = cv2.imread(r'D:\image\55c063e006b9ce3a87318cc919035afebf4324711ac2d-dFmBIA_fw480webp.webp')
image2 = cv2.resize(image2.copy(), image_weighted.shape[:2][::-1])
image3 = cv2.addWeighted(image_weighted.copy(), 0.4, image2, 0.6, 0)
image2_1 = (image2.copy() * 0.5).astype('uint8')
image_weighted_1 = (image_weighted.copy() * 0.5).astype('uint8')
image3_1 = (image2_1 * 0.8 + image_weighted_1 * 0.3 + 0).astype('uint8')
show(image3, 'image1')
