import numpy as np

array_list = np.array([
    1, 2, 3, 4, 5, 6, 7, 8])
np_arange = np.arange(10)
test_list = [1,2,3,4,5]
index = test_list.index(max(test_list))
test_list.insert(index,6)
print(test_list)
