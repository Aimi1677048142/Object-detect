import os

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from tool.operation import img_size, yolo_x_center_y_center_x_y, letter_box, show


class TargetDetectionDataSet(Dataset):

    def __init__(self, file_path: str, image_path: str, picture_suffix: str, file_suffix: str, device="cpu",
                 size=(640, 640)):
        self.file_path = file_path
        self.image_path = image_path
        self.device = device
        self.size = size
        self.image_path = image_path
        self.image_boxes_data = TargetDetectionDataSet.read_image_boxes(file_path, image_path, picture_suffix,
                                                                        file_suffix)

    def __getitem__(self, index):
        image, boxes = self.image_boxes_data[index]
        # 使用letter_box将图像进行缩放
        new_image, _, new_boxes, _, _, _, _ = letter_box(image, self.size, boxes)
        # 将数据进行转化为tensor
        torch_boxes = torch.from_numpy(new_boxes)
        torch_boxes = torch.cat([torch.ones(torch_boxes.shape[0], 1) * torch.inf, torch_boxes], dim=1)
        return torch.tensor(new_image).permute(2, 0, 1), torch_boxes

    def __len__(self):
        return len(self.image_boxes_data)

    @staticmethod
    def collate_fn(batch):
        image, boxes = zip(*batch)
        for i, b_box in enumerate(boxes):
            if len(b_box) > 0:
                b_box[..., 0] = i
        # 图像维度额转化
        image = torch.stack(image, dim=0)
        boxes = torch.stack(boxes, dim=0)
        return image, boxes

    @staticmethod
    def read_image_boxes(file_path: str, image_path: str, picture_suffix: str, file_suffix: str):
        image_boxes_list = []
        for file_obj in os.listdir(file_path):
            tmp_list = []
            file_name = os.path.splitext(file_obj)[0]
            if "classes" == file_name:
                continue
            width, height = img_size(os.path.join(image_path, file_name + picture_suffix))
            with open(os.path.join(file_path, file_name + file_suffix), mode='r', encoding='utf-8') as file:
                splitlines = file.read().splitlines()
                for split_line in splitlines:
                    split_list = split_line.split(' ')
                    split_list = [float(x) for x in split_list]
                    tmp_list.append(split_list)
            tmp_array = np.array(tmp_list)
            voc_array = yolo_x_center_y_center_x_y(tmp_array)
            image = cv2.imread(os.path.join(image_path, file_name + picture_suffix))
            voc_array[..., 1] = voc_array[..., 1] * width
            voc_array[..., 2] = voc_array[..., 2] * height
            voc_array[..., 3] = voc_array[..., 3] * width
            voc_array[..., 4] = voc_array[..., 4] * height
            image_boxes_list.append((image, voc_array.astype(int)))
        return image_boxes_list


file_path_txt = r'D:\pcb_data\original_ img and yolo_imfo\yolo_label_0902'
file_path_jpg = r'D:\pcb_data\original_ img and yolo_imfo\original img'
picture_suffix1 = ".jpg"
file_suffix1 = ".txt"
data_set = TargetDetectionDataSet(file_path_txt, file_path_jpg, picture_suffix1, file_suffix1)
train_data = DataLoader(
    dataset=data_set,
    batch_size=2,
    shuffle=True,
    collate_fn=TargetDetectionDataSet.collate_fn)

for img, box in train_data:
    img1 = torch.permute(img[0], (1, 2, 0)).numpy()
    show(img1, 'cc')
    print(box)
