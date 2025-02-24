# 第3章/PyTorchAPI/手写数字识别自定义训练.py
import torch
# pip install thop
# from thop import profile
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

"""
整个过程的步骤。记住
"""


# 0。搭建模型
class MyModel(torch.nn.Module):
    def __init__(self):
        # 继承Model类中的属性
        super(MyModel, self).__init__()
        # 全连接
        self.flat = torch.nn.Flatten()
        # Linear(input_channel,output_channel)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(784, 784),
            torch.nn.Sigmoid(),  # 激活层，网络层
            torch.nn.Linear(784, 128),
            # torch.nn.Sigmoid()
        )
        # 输出
        self.out = torch.nn.Linear(128, 10)

    def forward(self, input_x):
        x = self.flat(input_x)
        # 第1个网络层的计算和输出
        x = torch.sigmoid(self.fc(x))  # torch.sigmoid是一个数学函数
        return torch.sigmoid(self.out(x))


if __name__ == "__main__":
    model = MyModel()
    print("输出网络结构：", model)
    # PyTorch输入的维度是batch_size,channel,height,width
    # input = torch.randn(1, 1, 28, 28)
    # flops, params = profile(model, inputs=(input,))
    # print('浮点计算量：', flops)
    # print('参数量：', params)
    # 训练过程。
    # 1。加载数据集
    batch_size = 60
    learning_rate = 1e-5
    epochs = 10
    # 手写数字识别的数据集会自动下载到当前./data目录
    # 训练集
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            './data/',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    # 验证集
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            './data/',
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
        batch_size=batch_size, shuffle=True)
    # 2。设置优化器、学习率

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 3。设置损失函数
    criterion = torch.nn.CrossEntropyLoss()


    # 4。训练
    def model_train(data, label):
        # 清空模型的梯度
        model.zero_grad()
        # 前向传播
        outputs = model(data)
        # 计算损失
        loss = criterion(outputs, label)
        # 计算acc
        pred = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(pred == label) / label.shape[0]
        # 反向传播
        loss.backward()
        # 权重更新
        optimizer.step()
        # 进度条显示loss和acc
        desc = "train.[{}/{}] loss: {:.4f}, Acc: {:.2f}".format(
            epoch, epochs, loss.item(), accuracy.item()
        )
        return loss, accuracy, desc


    # 验证
    @torch.no_grad()
    def model_val(val_data, val_label):
        # 前向传播
        val_outputs = model(val_data)
        # 计算损失
        loss = criterion(val_outputs, val_label)
        # 计算前向传播结果与标签一致的数量
        val_pred = torch.argmax(val_outputs, dim=1)
        num_correct = torch.sum(val_pred == val_label)
        return loss, num_correct


    # 根据指定的epoch进行学习
    for epoch in range(1, epochs + 1):
        # 进度条
        process_bar = tqdm(train_loader, unit='step')
        # 开始训练模式
        model.train(True)
        # 因为train_loader是按batch_size划分的，所以都要进行学习
        for step, (data, label) in enumerate(process_bar):
            # 调用训练函数
            loss, accuracy, desc = model_train(data, label)
            # 输出日志
            process_bar.set_description(desc)
            # 在训练的最后1组数据后面进行验证
            if step == len(process_bar) - 1:
                # 用来计算总数
                total_loss, correct = 0, 0
                # 5。根据验证集进行验证
                model.eval()
                for _, (val_data, val_label) in enumerate(val_loader):
                    # 验证集前向传播
                    loss, num_correct = model_val(val_data, val_label)
                    total_loss += loss
                    correct += num_correct
                # 计算总测试的平均acc
                val_acc = correct / (batch_size * len(val_loader))
                # 计算总测试的平均loss
                val_Loss = total_loss / len(val_loader)
                # 验证集的日志
                var_desc = " val.[{}/{}]loss: {:.4f}, Acc: {:.2f}".format(
                    epoch, epochs, val_Loss.item(), val_acc.item()
                )
                # 显示训练和验证集的日志
                process_bar.set_description(desc + var_desc)
        # 进度条结束
        process_bar.close()
        # 保存模型和权重
        torch.save(model, './torch_mnist.pt')
