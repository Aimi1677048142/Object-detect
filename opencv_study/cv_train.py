import cv2


def show(image, title, is_debug=True, is_scale=False):
    if is_scale:
        img_h, img_w = image.shape[0:2]
        print(img_h, img_w)
        image = cv2.resize(image, [int(img_h * 0.5), int(img_w * 0.5)])
    if is_debug:
        cv2.imshow(title, image)
        cv2.waitKey(0)


def replace_picture(image1_h_index, image1_w_index, image2_h_index, image2_w_index, image1, image2, size, flag):
    """
    替换图片
    :param image1_h_index:
    :param image1_w_index:
    :param image2_h_index:
    :param image2_w_index:
    :param image1:
    :param image2:
    :param flag:
    :param size:
    :return:
    """
    image1_h, image1_w = image1.shape[:2]
    image2_h, image2_w = image2.shape[:2]
    if (image1_h_index + size > image2_h
            or image1_w_index + size > image1_w
            or image2_h_index + size > image2_h
            or image2_w_index + size > image2_w):
        return image1
    cropped_image = image2[image2_h_index:size + image2_h_index, image2_w_index:size + image2_w_index]
    image_new = cv2.addWeighted(
        image1[image1_h_index:size + image1_h_index, image1_w_index:size + image1_w_index],
        0.5, cropped_image, 0.4, 0)
    image1[image1_h_index:size + image1_h_index, image1_w_index:size + image1_w_index] = image_new
    if not flag:
        image1[image1_h_index:size + image1_h_index, image1_w_index:size + image1_w_index] = cropped_image
    return image1


image_1 = cv2.imread(r'D:\image\beauty-8870258_1280.webp')
image_2 = cv2.imread(r'D:\image\55c063e006b9ce3a87318cc919035afebf4324711ac2d-dFmBIA_fw480webp.webp')
show(replace_picture(
    50,
    50,
    100,
    100,
    image_1,
    image_2,
    100,
    True),
    '1')
