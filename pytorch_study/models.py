"""
训练的步骤：
1. 搭建模型结构
2. 设置优化器或者损失函数
3. 读取数据集
4. 反向传播
5. 评价模型

推理步骤：
1。加载训练好的weights
2。加载模型结构
3。进行前向传播
4。解析结果
"""
import torch
import torch.nn as nn


# 1. 继承nn.Module
# 2. 在__init__中构建算子的对象
# 3. 在forward()中执行前向传播
# 搭建models 时就初始了 weights，喂入数据时才执行forward
class MyModel(nn.Module):
    def __init__(self, in_features):
        """ 初始化模型对象"""
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(in_features, 784)
        self.fc2 = nn.Linear(784, 784)
        self.fc3 = nn.Linear(784, 784)
        self.out = nn.Linear(784, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.out(x)


class MyModel2(nn.Module):
    def __init__(self, in_features):
        super(MyModel2, self).__init__()
        # 序列化网络层
        self.fc = nn.Sequential(
            nn.Linear(in_features, 784),
            nn.Linear(784, 784),
            nn.Linear(784, 784),
            nn.Linear(784, 10)
        )

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = MyModel2(1 * 28 * 28).to(device)  # 类的初始，只会调__init__
    print(model)
    # 喂进去的数据，Batch x features
    x = torch.rand(30, 28 * 28).to(device)
    y = model(x)  # 前向传播
    print(y.shape)
