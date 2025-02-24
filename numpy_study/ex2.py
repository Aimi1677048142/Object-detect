"""
学习方法： 找到路线-》找到从0开始实战的文档-》先模仿做起来-》先做小demo(案例)-》再回来梳理理论
         实践-->总结
         善于利用互联网
         1. 官方帮助文档 numpy pytorch
         2. github
         3. google
         4. stackoverflow.com(csdn 抄)
         5. 博客园
"""
import numpy as np

ll = [
    [
        [
            [1, 2, 3],
            [2, 3, 4]
        ],
        [
            [1, 2, 3],
            [2, 3, 4]
        ]
    ],
    [
        [
            [1, 2, 3],
            [2, 3, 4]
        ],
        [
            [1, 2, 3],
            [2, 3, 4]
        ]
    ],
    [
        [
            [1, 2, 3],
            [2, 3, 4]
        ],
        [
            [1, 2, 3],
            [2, 3, 4]
        ]
    ],
]

# 1.数据类型如何定义（如何转换）
arr = np.array(ll, dtype=np.float32)
print(arr.shape)
# 0 - 255
# /255.0 ->[0 ,1] 归一化

arr1 = np.zeros([240, 640, 3], dtype=np.float32)  # 默认是float64 [numpy(float64) ->pytorch(float64)] pytorch默认是float32
print(arr.shape)
arr2 = np.ones([640, 640, 3], dtype=np.float32)
arr3 = np.ones_like(arr1)  # shape像arr1
# range()只能int; arange()可以float
# ValueError: cannot reshape array of size 50 into shape (1,1,1,20)
# size总数要相等.
arr4 = np.arange(0, 10, 0.2).reshape([5, 10])
arr5 = np.linspace(0, 2, 100).reshape([10, 10])  # 均分，得到1维
arr6 = np.random.randn(28, 28, 3)  # *dn代表任意个参数，满足正态分布

# 2. 数学运算
#     +,-,*,/
a = np.array([[1, 2], [3, 4]])
b = np.array([[1], [2]])
print(a + b)
pass

# 3. 了解+,-,*,/  矩阵乘法

a = np.random.randn(2, 3)
b = np.random.randn(3, 3)

print((a @ b).shape)  # ****
print('#' * 10)
print(a.dot(b).shape)
print('#' * 10)
print(np.matmul(a, b).shape)  # ****
print('#' * 10)

a = np.random.randn(2, 3)
b = np.random.randn(2, 3)

print((a * b))
print(np.multiply(a, b))  # ****
