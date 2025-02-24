import numpy as np

# 创建一个 ndarray
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 保存到文本文件
np.savetxt('array_file.txt', arr, fmt='%d')

# 从文本文件中加载
loaded_arr = np.loadtxt('array_file.txt', dtype=int)
