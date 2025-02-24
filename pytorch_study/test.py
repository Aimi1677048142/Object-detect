import numpy as np
import torch

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64, requires_grad=True)
Y = torch.pow(x, 2)
Y.backward(gradient=torch.ones_like(Y))

a = torch.tensor([1, 2, 3], dtype=torch.float64)
b = torch.tensor([3, 4, 5], dtype=torch.float64)
# print(a + b)
# 生成一个网格矩阵
# t, m = torch.meshgrid(a, b)
# print(t, m)

g = torch.randn([4, 28, 28, 3])
r = torch.randn([4, 28, 28, 3])
print(torch.cat([g, r], dim=0).shape)
print(torch.reshape(g, [-1, 28, 3]).shape)
print(g.reshape([4, 28, 28, 3]).shape)
print(torch.unsqueeze(g, dim=0).shape)
torch_ones = torch.reshape(torch.ones(3)*torch.inf,[-1,1])
print(torch_ones)

# print(x.grad)
