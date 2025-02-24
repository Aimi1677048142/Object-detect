import torch
import numpy as np

a = torch.randn([4, 28, 28, 3])
b = torch.randn([2, 28, 28, 3])
c = torch.cat([a, b], dim=0)  # 6 x28x28x3
new_c = torch.reshape(c, [6 * 28, 28, 3])  # shape 改变
new_c2 = new_c.view([6, 28, 28, 3])
# 升维和降维
new_c3 = new_c2.unsqueeze(dim=0)  # 1x6x28x28x3
new_c4 = new_c3.squeeze(dim=0)  # 6x28x28x3，只能对shape为1进行降维
# pytorch 1次只能交换2个维度
new_c5 = new_c4.transpose(-1, -2) # 6x28x3x28
new_c6 = new_c5.transpose(0, 1) # 28 x6 x 3 x 28
# 1次将所有维度进行转置
n_transpose = np.random.randn(4, 28, 28, 3).transpose([-1,-2,1,0])
pass
