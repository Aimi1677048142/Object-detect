import os

import cv2
import torch
from torch.utils.data import Dataset

from data_augmentation import letterbox


class ClassificationDateSet(Dataset):
    def __init__(self, object_dir, class_mapping: dict, device="cpu"):
        self.object_dir = object_dir
        self.data, self.labels, self.filename_list = self.get_classification_by_dir(object_dir, class_mapping)
        self.device = device

    def __getitem__(self, index: int):
        # 使用 cv2 读取 JPG 图像
        sample = cv2.imread(self.data[index])
        if sample is None:
            raise ValueError(f"Image at index {index} could not be loaded: {self.data[index]}")
        sample, _, _, _, _, _, _ = letterbox(sample, size=(224, 224))
        # 转换为浮点型并归一化
        sample = torch.tensor(sample, dtype=torch.float32) / 255.0
        # 可能需要调整维度顺序
        sample = sample.permute(2, 0, 1)  # 将 HWC 转换为 CHW
        return sample, self.labels[index], self.filename_list[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_classification_by_dir(object_dir: str, class_mapping: dict):
        data_list = []
        label_list = []
        filename_list = []
        # 遍历指定目录及其所有子目录
        for dir_path, _, filenames in os.walk(object_dir):
            # 获取相对于根目录的相对路径
            relative_path = os.path.relpath(dir_path, object_dir)
            # 如果当前路径是根目录 itself, 跳过它
            if relative_path == ".":
                continue
            absolute_paths = [os.path.join(dir_path, filename) for filename in filenames]
            filename_list.extend(filenames)
            data_list.extend(absolute_paths)
            num = class_mapping.get(relative_path)
            label_list.extend([num] * len(filenames))
        return data_list, label_list, filename_list

# root_dir = r'D:\pcb_data\train_data'
# date_set = ClassificationDateSet(root_dir)
# # print(len(date_set))
# train_data = DataLoader(
#     dataset=date_set,
#     batch_size=2,
#     shuffle=True
# )
# for data, label in train_data:
#     pass
