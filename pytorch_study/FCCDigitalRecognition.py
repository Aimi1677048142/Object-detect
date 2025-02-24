import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from FCCDigitalDataSet import FCCDigitalDataSet


class FCCDigitalRecognition(nn.Module):
    def __init__(self, in_features: int, hidden1_features: int, hidden2_features: int, hidden3_features: int,
                 out_features: int):
        super(FCCDigitalRecognition, self).__init__()
        # 摊平数据
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden1_features),
            nn.ReLU(),
            nn.Linear(hidden1_features, hidden2_features),
            nn.ReLU(),
            nn.Linear(hidden2_features, hidden3_features)
        )
        self.out = nn.Linear(hidden3_features, out_features)

    def forward(self, input_x):
        x = self.flatten(input_x)
        # 为什么跟上面不一样呢,这是因为上面的是激活函数模型层,torch.relu是一个数学函数
        x = torch.relu(self.fc(x))
        return torch.softmax(self.out(x), dim=1)


model = FCCDigitalRecognition(1024, 1024, 1024, 128, 10)
# 初始化参数
batch_size, learning_rate, epochs = 20, 1e-4, 100
# 加载数据集
img_train_path, img_val_path = r'D:\training_img', r'D:\test_img'
data_train_set, data_val_set = FCCDigitalDataSet(img_train_path), FCCDigitalDataSet(img_val_path)
# 测试数据集
# train_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True),
#     batch_size=batch_size, shuffle=True)
train_loader = DataLoader(
    dataset=data_train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=FCCDigitalDataSet.collate_fn
)

# 验证数据集
# val_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True),
#     batch_size=batch_size, shuffle=True)
val_loader = DataLoader(
    dataset=data_val_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=FCCDigitalDataSet.collate_fn
)
# 设置优化器,学习率
optim_adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 设置损失函数
criterion = nn.CrossEntropyLoss()


def model_train(data, label, epoch):
    # 清空模型的梯度
    model.zero_grad()
    # 前向传播
    out_puts = model(data)
    # 计算损失
    loss = criterion(out_puts, label)
    # 推断
    predict = torch.argmax(out_puts, dim=1)
    # 计算精度
    accuracy = torch.sum(predict == label) / label.shape[0]
    # 反向传播
    loss.backward()
    # 权重更新
    optim_adam.step()
    # 进度条显示loss和acc
    desc = "train.[{}/{}] loss:{:.4f}, Acc:{:.2f}".format(epoch, epochs, loss, accuracy)
    return loss, accuracy, desc


@torch.no_grad
def model_val(data, val_label):
    # 前向传播
    out_puts = model(data)
    # 计算损失
    loss = criterion(out_puts, val_label)
    # 计算前向传播结果与标签一致的数量
    val_puts = torch.argmax(out_puts, dim=1)
    predict_num = torch.sum(val_puts == val_label)
    return loss, predict_num


for epoch in range(1, epochs + 1):
    process_bar = tqdm(train_loader, unit='step')
    # 开始训练
    model.train(True)
    for step, (train_data, train_label) in enumerate(process_bar):
        # 调用训练函数进行训练
        # train_label = nn.functional.one_hot(train_label,10)
        loss, accuracy, desc = model_train(train_data, train_label, epoch)
        # 输出日志
        process_bar.set_description(desc)
        if step == len(process_bar) - 1:
            total_loss, total_correct = 0, 0
            model.eval()
            for _, (val_data, label) in enumerate(val_loader):
                # label = nn.functional.one_hot(label, 10)
                val_loss, val_predict_num = model_val(val_data, label)
                total_loss += val_loss
                total_correct += val_predict_num
            val_total_mean_correct = total_correct / (batch_size * len(val_loader))
            val_total_mean_loss = total_loss / len(val_loader)
            # 验证集的日志
            val_desc = "val.[{}/{}] loss:{:.4f}, Acc:{:.2f}".format(epoch, epochs, val_total_mean_loss.item(),
                                                                    val_total_mean_correct.item())
            process_bar.set_description(desc + val_desc)
    process_bar.close()
    torch.save(model, './torch_mnist1.pt')
