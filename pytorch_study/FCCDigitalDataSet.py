import os

import torch
import cv2
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class FCCDigitalDataSet(Dataset):
    def __init__(self, image_path: str, device="cpu"):
        self.image_path = image_path
        self.device = device
        self.image_data, self.targets = FCCDigitalDataSet.get_digital_data(image_path)
        self.image_data = np.array(self.image_data)  # 转换为 NumPy 数组
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        image_data, image_label = self.image_data[index], self.targets[index]
        image_pil = Image.fromarray(image_data, mode="L")
        transform = transforms.ToTensor()
        img_tensor = transform(image_pil)
        return img_tensor, image_label

    def __len__(self):
        return len(self.image_data)

    @staticmethod
    def collate_fn(batch):
        image, labels = zip(*batch)
        # 图像维度额转化
        image = torch.stack(image, dim=0)
        labels = torch.tensor(labels)
        return image, labels

    @classmethod
    def get_digital_data(cls, image_path):
        image_result, image_label_list = [], []
        for image in os.listdir(image_path):
            num = int(image.split("_")[0])
            img = cv2.imread(os.path.join(image_path, image), 0)
            if img is None:
                continue
            img = img / 255.0
            image_result.append(img)
            image_label_list.append(num)
        return image_result, image_label_list


# img_path = r'D:\training_img'
# data_set = FCCDigitalDataSet(img_path)
# train_data = DataLoader(
#     dataset=data_set,
#     batch_size=2,
#     shuffle=True,
#     collate_fn=FCCDigitalDataSet.collate_fn
# )
#
# train_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True),
#     batch_size=2, shuffle=True)
# for data, label in train_loader:
#     pass
