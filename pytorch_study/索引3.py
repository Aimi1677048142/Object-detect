import torch
import numpy as np
# 口诀：在那个维度取值，那个维度就相对应的去掉，或者减少! 按逗号进行维度匹配。

# 索引切片的语法与numpy保持一致
a = torch.randn([4, 28, 28, 3])
print(a[[0, 1], ...].shape)
print(a[0, 2, :].shape)

print(a[:, :, :, 2].shape)
print(a[:, 0, :, :].shape)

# tensor支持直接修改，tensorflow里面tesnor是不能直接修改的
a[:, :, :, 0] = torch.randn(4, 28, 28)  # 跟numpy一致，修改的时候维度要一致

# 与numpy一样，也支持布尔取值
data = torch.tensor([[1,2],[3,4],[5,6]]) # 3x2
mask = data > 3
new_data = data[mask] # 只能有1个条件

print(a[[0,1],[2,3],[4,5],:].shape)
pass
