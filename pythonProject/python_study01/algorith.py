# 二分法查找
def half_algorith(nums_array: list, target_num: int) -> int:
    i, j = 0, len(nums_array) - 1
    while i <= j:
        m = (i + j) // 2
        if nums_array[m] < target_num:
            i = m + 1
        elif nums_array[m] > target_num:
            j = m - 1
        else:
            return m
    return -1


# 二分法插入值的索引,无重复的列表
def half_algorith_insert_not_repeat(nums1: list, target1: int) -> int:
    i, j = 0, len(nums1) - 1
    while i <= j:
        m = i + j // 2
        if nums1[m] < target1:
            i = m + 1
        elif nums1[m] > target1:
            j = m - 1
        else:
            return m
    return i


# 二分法插入值的索引，重复的列表
def half_algorith_insert_repeat(nums2: list, target2: int) -> int:
    i, j = 0, len(nums2) - 1
    while i <= j:
        m = i + j // 2
        if nums2[m] < target2:
            i = m + 1
        elif nums2[m] > target2:
            j = m - 1
        else:
            j = m - 1
    return i


# 二分查找左边界
def binary_search_left_edge(array: list[int], target_num: int) -> int:
    i = half_algorith_insert_repeat(array, target_num)
    if array[i] != target_num or i == len(array):
        return -1
    return i


# 二分查找右边界
def binary_search_right_edge(arr: list[int], target_n: int) -> int:
    i = half_algorith_insert_repeat(arr, target_n + 1)
    j = i-1
    if arr[j] != target_n or j == -1:
        return -1
    return j


nums, target = [1, 2, 3, 4, 5, 6, 7], 5
result_num = half_algorith(nums, target)
print(result_num)
