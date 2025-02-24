# 任务：读取D:\DLAI\qwenAILearn\VGGSSD\facemask_detection-main\face_mask\2021_train_yolo.txt
# 将里面的图片和BOX，按batchsize返回
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch

facePath = r"D:\pcb_data\face_mask\2021_train_yolo.txt"


# 1. 继续 dataset类
#    __init__ 初始对象
#    __getitem__  随机取某条数据
#    __len__      总数量
#    collate_fn【不是必须，一般都有】  按batch压缩数据返回，重写
# 2. dataset生成器对象，传给dataloader这个迭代器，按指定batch在自定义的collate_fn返回数据,x/y

class FaceDataSet(Dataset):
    # 1.初始化
    def __init__(self, device="cpu", size=(640, 640), path=facePath):
        self.txt_path = path
        self.device = device
        self.size = size
        self.data = self.read_txt()  # 2.从txt读自定义文件的目录

    # 3.返回文件列表中的第几个
    def __getitem__(self, item):
        """
        随机获取某条数据
        :param item:
        :return:
        """
        row_data = self.data[item]
        return self.get_data_rows(row_data)

    # 4.自定义解析图片和label的方式
    def get_data_rows(self, row):
        """
        'D:/DLAI/qwenAILearn/VGGSSD/facemask_detection-main/face_mask/facemask_dataset/19_Couple_Couple_19_461.jpg 441,432,525,528,0 592,435,676,546,0
        :param row:
        :return:
        """
        row1 = row.strip()
        # 观察txt分割成多个部分的规律
        data = row1.split(" ")
        # 读图片
        img = cv2.imread(data[0])  # bgr
        img = cv2.resize(img, self.size)  # **** 更换为等比例缩放的函数。opencv resize函数，非常占用性能。正式用的不推荐。
        # result = []
        # for d in data[1:]:
        #     detail_data = d.split(',')
        #     result.append(detail_data)
        # box->xyxy->yolo格式 [0](batchSize中的第几张图torch.inf) +[1,2,3,4,0]
        # 这一段代码其实实在调用dataloader的时候,每次批量取图片是哪一张图片的
        boxes = torch.tensor([np.array([torch.inf] + d.split(','), dtype=np.float32) for d in data[1:]],
                             device=self.device, dtype=torch.float32)
        # 图片的归一化 img/255.
        # boxes的归一化/ heigh, /width
        # cv:height,width,3 -> 3,height,width
        return torch.tensor(img / 255.).transpose(-1, 0), boxes

    # 求总共数据的长度
    def __len__(self):
        return len(self.data)

    # 6. 自定义返回 实现label的拼接，并且知道是第几个图
    #
    @staticmethod
    def collate_fn(batch):
        img, box = zip(*batch)

        for i, l in enumerate(box):
            if len(l) > 0:
                l[..., 0] = i
        img = torch.stack(img, 0)  # 维度都是一样. 2x 3x640x640
        last_box = torch.cat(box, 0)  # 只能某一个维度不一样。拼接(相加,cat,add)
        return img, last_box

    def read_txt(self):
        """
        读取txt里面的所有内容
        :return:
        """
        with open(self.txt_path, encoding='utf-8') as f:
            rows = f.readlines()
            return rows


if __name__ == "__main__":
    t = FaceDataSet(path=facePath)  # 自定义的迭代器
    # list: 1->640x640x3   BOX1个 1x5 --> 1x6  0,123,234,543,345,0
    # list: 2->640x640x3   BOX2个 2x5 --> 2x6  1,123,234,543,345,0  1,123,234,543,345,0

    #      2x640x640x3    3x5
    # pytorch BCHW    tensorflow    BHWC

    # DataLoader是生成器
    train_data = DataLoader(
        dataset=t,
        batch_size=2,
        shuffle=True,
        collate_fn=FaceDataSet.collate_fn  # 按指定的batch的某种方式进行返回 回调函数 我们用那种生成器返回
    )
    for image, label in train_data:
        pass
