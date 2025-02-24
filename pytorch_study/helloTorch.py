import torch
import numpy as np

# pytorch默认float32. numpy默认是float64
# device = "cuda:0" if torch.cuda.is_available() else 'cpu'
# # 将list转换成tensor
# dd = torch.tensor([1.0, 2.0, 3.0], device=device)
# print(dd)
# # pass
#
# X = torch.linspace(0, 1, 5)
# Y = torch.linspace(10, 20, 5)
# x, y = torch.meshgrid([X, Y])
# pass

# 自动求导

# 创建一个需要梯度的张量
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# 计算每个元素的平方
b = torch.pow(x, 2)

# 创建一个与 b 形状相同的梯度张量，并传递给 backward()
gradients = torch.ones_like(b)
b.backward(gradient=gradients)
print(len(b))

# 输出 x 的梯度
print(x.grad)
