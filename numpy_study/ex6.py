import numpy as np

# 广播机制的好处
# 在本地存储的时候，占用空间较小。但是计算的时候，会自动扩展，方便计算

x = np.random.random([2, 4])
w = np.random.random([4, 3])
b = np.array([3])
#        2x3 +
result = x @ w + b # b 自动扩维为2 x 3的数组，并且内容为3
print(result.shape)

# 并不是所有数组都可以广播，需要满足以下规则：
# 从后往前，要么为shape为1，要么相等
# shape
# b1,h1,w1,c1
#           1
#        w1,1
#     h1,w1,1
#     h1,1,1
#     h1,1,c1
#  1  1,1,1
#  b1 1,1,1
#  b1 h1,1,1

# 32, 32, 2
# 32, 1 最后一个维度为 1 可以广播
x = np.random.random([32, 32, 2])
x = x + np.random.random([32, 1])
print(x.shape)
print('##########################')
# 32, 32, 2
# 32, 2 最后二个给度相同可以广播
x = np.random.random([32, 32, 2])
x = x + np.random.random([32, 2])
print(x.shape)
print('##########################')
# 32, 32, 2
# 32, 1 最后一个维度为 1 可以广播
x = np.random.random([32, 32, 2])
x = x + np.random.random([32, 1])
print(x.shape)
print('##########################')
# 32, 32, 2
# 1 , 32, 1 两边为 1，中间为 32 相等，可以广播
x = np.random.random([32, 32, 2])
x = x + np.random.random([1, 32, 1])
print(x.shape)
print('##########################')
# 32, 32, 2
# 3, 2 倒数第 2 个维度不等，不能广播
x = np.random.random([32, 32, 2])
x = x + np.random.random([3, 2])
print(x.shape)

"""
1. 如何初始
2. 切片
3. 花式索引
4. 广播
"""