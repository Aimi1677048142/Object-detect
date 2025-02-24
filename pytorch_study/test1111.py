import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# from data_augmentation import letterbox


class ClassificationDateSet(Dataset):
    # 初始化数据集，接受一个对象目录
    def __init__(self, object_dir, device="cpu"):
        self.object_dir = object_dir
        # 通过目录获取数据和标签
        self.data, self.labels = self.get_classification_by_dir(object_dir)
        self.device = device

    # 获取指定索引的样本
    def __getitem__(self, index: int):
        # 使用 cv2 读取 JPG 图像
        sample = cv2.imread(self.data[index])
        if sample is None:
            raise ValueError(f"Image at index {index} could not be loaded: {self.data[index]}")
        # sample, _, _, _, _, _, _ = letterbox(sample, size=(224, 224))
        # 转换为浮点型并归一化
        sample = torch.tensor(sample, dtype=torch.float32) / 255.0
        # 可能需要调整维度顺序
        sample = sample.permute(2, 0, 1)  # 将 HWC 转换为 CHW
        return sample, self.labels[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_classification_by_dir(object_dir: str):
        dir_dict = {}  # 用于存储目录和文件名的字典
        data_list = []  # 存储所有图像的绝对路径
        label_list = []
        # 用于存储每个目录的文件数量
        dir_file_count = {}

        # 遍历指定目录及其所有子目录
        for dir_path, _, filenames in os.walk(object_dir):
            # 获取当前目录dir_path 相对于 根目录object_dir的相对路径
            relative_path = os.path.relpath(dir_path, object_dir)
            # 如果当前路径是根目录, 跳过
            if relative_path == ".":
                continue
            # 将相对路径作为字典的键，文件列表作为值{相对路径：文件列表}
            dir_dict[relative_path] = filenames
            # 统计每个目录的文件数量
            if 1000 < len(filenames):
                dir_file_count[relative_path] = len(filenames)
            # 生成绝对路径列表
            absolute_paths = [os.path.join(dir_path, filename) for filename in filenames]
            data_list.extend(absolute_paths)
            num = 1
            # if num > 53:
            #     num -= 1

            label_list.extend([num] * len(filenames))
        print(dir_file_count)
        # 绘制文件数量的分布图
        plt.figure(figsize=(12, 8))
        plt.barh(list(dir_file_count.keys()), list(dir_file_count.values()), color='skyblue')
        plt.xlabel('文件数量')
        plt.title('每个子目录的文件数量分布')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        return data_list, label_list


root_dir = r'D:\pcb_data\new_bird\new_bird'
date_set = ClassificationDateSet(root_dir)
# print(len(date_set))
# train_data = DataLoader(
#     dataset=date_set,
#     batch_size=2,
#     shuffle=True
# )
# for data, label in train_data:
#     pass