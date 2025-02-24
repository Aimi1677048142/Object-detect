# 选择排序
import time


def select_sort(nums: list[int]) -> list[int]:
    n = len(nums)
    for i in range(n - 1):
        k = i
        for j in range(k + 1, n):
            if nums[j] < nums[k]:
                k = j
        nums[i], nums[k] = nums[k], nums[i]
    return nums


# 冒泡排序
def compare_sort(nums: list[int]) -> list[int]:
    for i in range(len(nums) - 1, 0, -1):
        for j in range(i):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    return nums


# 选择排序算法
def select_sort(nums: list[int]) -> list[int]:
    for i in range(len(nums) - 1, 0, -1):
        position_max = 0
        for j in range(1, i + 1):
            if nums[j] > nums[position_max]:
                position_max = j
        nums[i], nums[position_max] = nums[position_max], nums[i]


# 插入排序算法


def insertion_sort(nums: list[int]) -> list[int]:
    for i in range(1, len(nums)):
        current_value = nums[i]
        position = i
        while position > 0 and nums[position - 1] > current_value:
            nums[position] = nums[position - 1]
            position -= 1
        nums[position] = current_value
    return nums


print(insertion_sort([1, 3, 4, 2, 5]))


# 快速排序
# 先找哨兵划分
def partition(nums: list[int], left: int, right: int):
    i, j = left, right
    while i < j:
        while i < j and nums[j] >= nums[left]:
            j -= 1
        while i < j and nums[i] <= nums[left]:
            i += 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i], nums[left] = nums[left], nums[i]
    return i


# 分片递归
def quick_sort(nums: list[int], left: int, right: int):
    if left >= right:
        return
    pivot = partition(nums, left, right)

    quick_sort(nums, left, pivot - 1)
    quick_sort(nums, pivot + 1, right)


# 尾递归优化
def quick_sort_last(nums: list[int], left: int, right: int):
    while left < right:
        pivot = partition(nums, left, right)
        if pivot - left < right - pivot:
            quick_sort(nums, left, pivot - 1)
            left = pivot + 1
        else:
            quick_sort(nums, pivot + 1, right)
            right = pivot - 1


# 归并排序：无限制的拆分数据，做合并
def merge_sort(nums: list[int]):
    # 递归结束条件
    if len(nums) <= 1:
        return nums
    # 创建一个中间值
    mid = len(nums) // 2
    left = nums[:mid]
    right = nums[mid:]
    merge_sort(left)
    merge_sort(right)
    merge_list = []
    while left and right:
        if left[0] < right[0]:
            merge_list.append(left.pop(0))
        else:
            merge_list.append(right.pop(0))
    merge_list.extend(right if right else left)
    return merge_list


# 堆排序
def shift(li, low, high):
    """

    :param li:
    :param low: 堆的开始根节点
    :param high: 堆的最大叶子节点
    :return:
    """
    i = low
    j = 2 * i + 1
    tmp = li[low]

    while j <= high:
        if j + 1 <= high and li[j] < li[j + 1]:
            j = j + 1
        if li[j] > tmp:
            li[i] = li[j]
            i = j
            j = 2 * i + 1
        else:
            li[i] = tmp
            break
    else:
        li[i] = tmp


def heap_sort(li):
    n = len(li)
    # 构建堆的过程
    for i in range((n - 2) // 2, -1, -1):
        shift(li, i, n - 1)

    for i in range(n - 1, -1, -1):
        li[0], li[i] = li[i], li[0]
        shift(li, 0, i - 1)
    return li


li = [x for x in range(100)]
import random

random.shuffle(li)


# print(li)

# sort = heap_sort(li)
# print(sort)


# 桶排序
def buck_sort(li: list[int], num):
    buck_num = (max(li) - min(li)) // num + 1
    buck_list = [[] for i in range(buck_num)]
    for i in li:
        buck_list[i // num].append(i)
    for obj in buck_list:
        heap_sort(obj)
    li.clear()
    for obj in buck_list:
        li.extend(obj)
    return li


print(buck_sort(li, 40))


# 计数排序
def count_sort(nums: list[int]) -> list:
    if len(nums) < 2:
        return nums
    max_nums = max(nums)
    count_list = [0] * (max_nums + 1)
    for num in nums:
        count_list[num] += 1
    nums.clear()
    for index, value in enumerate(count_list):
        nums.extend(value * [index])
    return nums


# 基数排序
# 硬币找零递归问题
def recDC(coins_value_list, change, known_results):
    min_coins = change
    if change in coins_value_list:
        known_results[change] = 1
        return 1
    elif known_results[change] > 0:
        return known_results[change]
    else:
        for i in [c for c in coins_value_list if c <= change]:
            nums_coins = 1 + recDC(coins_value_list, change - i, known_results)
            if nums_coins < min_coins:
                min_coins = nums_coins
                known_results[change] = min_coins
    return min_coins


start = time.process_time()
print(recDC([1, 5, 10, 25], 63, [0] * 64))
end = time.process_time()
print(end - start)


# 硬币兑换，动态规划求解
def dp_make_change(coins_value_list, change, tmp_list, coins_user):
    for coin in range(1, change + 1):
        min_coins = coin
        new_coins = 1
        for j in [c for c in coins_value_list if c <= change]:
            if tmp_list[coin - j] + 1 < min_coins:
                min_coins = tmp_list[coin - j] + 1
                new_coins = j
        tmp_list[coin] = min_coins
        coins_user[coin] = new_coins
    return tmp_list[change],coins_user


def print_coins(coins_user, change):
    current_coins = change
    while current_coins > 0:
        this_coins = coins_user[current_coins]
        print(this_coins)
        current_coins -= this_coins
    return coins_user


result1,conis_user = dp_make_change([1, 5, 10, 21, 25], 63, [0] * 64, [0] * 64)

print(print_coins(conis_user, 63))
