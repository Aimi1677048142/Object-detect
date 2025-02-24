import numpy as np

# 向下取整数
a = np.floor([3.14, 3.345, 4.89])
print(a)

# 向上取整
b = np.ceil([3.14, 3.345, 4.89])
print(b)

num2 = np.random.randn(2, 3)
print(num2)
# 转置. 行列交换
print(num2.T)
print(np.transpose(num2, axes=[-1, 0]))

print('#' * 100)
num3 = np.random.randn(1, 3)
# 堆叠在一起
print(np.concatenate([num2, num3], axis=0))  # 除了当前维度可以不一样，其它维度要一样

# 维度变换
num4 = np.random.randn(4, 28, 28, 3)
num5 = num4.copy()
num6 = num4.copy()
num4 = num4[:, np.newaxis, np.newaxis, ...]  # 升维的方法
print(num4.shape)
print(num5[:, None, None, ...].shape)
num7 = np.expand_dims(num6, axis=0)
print(num7.shape)  # 1x4x28x28x3
print(num7.squeeze(axis=0).shape)  # 降维，只能降1
print(num7.reshape([-1, 28]).shape)  # -1自动推断，只能有1个
print(num4.reshape([-1, 28]).shape)
# 通函数
print(np.sqrt(2))
print(np.exp(3))
print(np.mean(num2))
