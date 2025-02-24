import numpy as np

num1 = np.arange(12)
print(num1)
print("*" * 100)
i = np.array([1, 1, 3, 8, 5])
print(num1[i])  # 如果是1维tensor，i取对应num1的下标的值
print("*" * 100)
j = np.array([[3, 4], [9, 7]])
print(num1[j])  # 还是按1维tensor取值，j取完之后组成一个2维
print("*" * 100)
# 5x3: wc
palette = np.array([[0, 0, 0],  # black
                    [255, 0, 0],  # blue
                    [0, 255, 0],  # green
                    [0, 0, 255],  # red
                    [255, 255, 255]])  # white
image = np.array([[0, 1, 2, 0],
                  [0, 3, 4, 0]])
print(palette[image])  # 按axis=0轴，按image中的下标，重复取值。取了之后升维
print("#" * 100)
num2 = np.arange(12).reshape(3, 4)
print(num2)
i = np.array([[0, 1, 2],
              [1, 2, 2]])
j = np.array([[2, 1, 0],
              [3, 3, 0]])

print(num2[i, j])  # 2个维度同时取. 取i的同时要取j，要size对应
print('#' * 100)
print(num2[i, 2])  # 每一个i，同时取2
print('#' * 100)
print(num2[:, j])  # 所有行，重复取j

print('#' * 100)
# 布尔取值
num3 = np.arange(12).reshape(3, 4)
print(num3)
print('#' * 100)
# print(num3>2 and num3<5)
print(num3 > 2)  # 只能有一个布尔值
print(num3[num3 > 2])  # 取为true，false的会丢弃，变成1维
print('*' * 100)
b1 = np.array([1, 2])
b2 = np.array([0, 2])
"""
b1 = np.array([False,True,True])             # first dim selection
b2 = np.array([True,False,True,False])
"""
print(num3[b1])
print('*' * 100)
print(num3[:, b2])
print('>' * 100)
print(num3[b1, b2]) # b1先取, b2的时取得是b1 and b2的结果
