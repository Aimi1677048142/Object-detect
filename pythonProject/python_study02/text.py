import copy
import math
import random

import numpy

for i in range(100, 1000):
    if i == (i // 100) ** 3 + (i // 10 % 10) ** 3 + (i % 10) ** 3:
        print(i)


def trand(n: int):
    k = n - 1
    for i in range(0, n):
        for j in range(k):
            print(end=' ')
        k = k - 1
        for j in range(0, i + 1):
            print('* ', end='')

        print("\r")


trand(4)

# 生成九九乘法表
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f"{j}x{i}={j * i}", end="\t")
    print()  # 换行

rand_int = random.randint(1, 100)

# while True:
#     input_num = int(input("请输入一个数："))
#     if input_num > rand_int:
#         print("猜大了")
#     elif input_num < rand_int:
#         print("猜小了")
#     else:
#         print("猜对了")
#         break
#
#     print('错误')

# print('正确') if len(pwd := input('请设置密码')) == '123456' else print('输入长度不足 6 位') if len(pwd) < 6 else print(
#     '错误')


s10 = {1, 2, 3, 4, 5}
s = s10.copy()
copy_deepcopy = copy.deepcopy(s10)
s10.remove(2)
print(s, copy_deepcopy, id(s10), id(s), id(copy_deepcopy))

print(f'{3}{4}{5}')

arr1 = [[2, 5, 3, 6, 3], [1, 3, 5, 1, 8], [2, 5, 9, 2, 8]]
dct1 = {}
# for obj in arr1:
#     for one in obj:
#         if one in dct1:
#             dct1[one] += 1
#         else:
#             dct1[one] = 1
# print(dct1)
[None for obj in arr1 for one in obj if dct1.update({one: dct1.get(one, 0) + 1}) is None]
print(dct1)

# input_num=int(input("请输入数"))
# print(f'{input_num}不是质数') if input_num <= 1 or any([True if input_num % i == 0 else False for i in range(2,input_num)]) else print(f'{input_num}是质数')
# print([x for x in range(1, 101) if
#        (True if x <= 1 or any([False if x % i == 0 else False for i in range(2, x)]) else True)])

print([x for x in range(100, 1000) if int(str(x)[0]) ** 3 + int(str(x)[1]) ** 3 + int(str(x)[2]) ** 3 == x])
# 示例数据
user_info = [
    {'name': 'yoyo', 'pwd': '123456'},
    {'name': 'haha', 'pwd': 'qwe123'},
    {'name': 'xiaoli', 'pwd': 'qwe123!'},
    {'name': 'xiaowang', 'pwd': None}
]


# 定义一个函数来评估密码强度
def password_strength(pwd):
    if not pwd:
        return 0  # 无密码或密码为None，强度为0
    strength = ""
    if any(c.isdigit() for c in pwd):
        strength = "弱"
    if any(c.isalpha() for c in pwd):
        strength = '中'
    if any(not c.isalnum() for c in pwd):
        strength = '强'
    return strength


user_strength = {user['name']: password_strength(user['pwd']) for user in user_info if user['pwd'] is not None}

print(user_strength)
lst12 = [x for x in range(1, 1001) if (sum(k for k in range(1, x) if x % k == 0) == x)]
print(lst12)


def ce_sum(*args, y):
    if len(args) == 0:
        return 0
    sum1 = 0
    for index, v in enumerate(args):
        sum1 += (index + 1) * v
    return sum1 * y


def ce_sum1(*args, **kwargs):
    return sum([(i + 1) * v for i, v in enumerate(args)]) * tuple(kwargs.values())[0]


print(ce_sum(1, 2, 3, 4, y=5))

number = 0
account = "abc"


def deposits_withdrawals(account, num, flag):
    global number
    if account != 'abc':
        return
    if account == 'abc' and flag:
        number += num

    elif account == 'abc' and not flag:
        number -= num
        if number < 0:
            number += num
    else:
        return
    return number


# while account == 'abc':
#     deposits_withdrawals_name = input("请输入存款还是取款")
#     num = int(input("请输入金额"))
#     deposits_withdrawals(account, num, True if deposits_withdrawals_name == '存款' else False)
#     print(f"剩余金额{number}")


def generate_sequence(n):
    sequence = [1, 4, 2, 8, 6]  # 初始化序列的前5个数字
    while len(sequence) < n:
        if len(sequence) % 2 == 0:  # 偶数位置
            next_value = sequence[-1] * 4
        else:  # 奇数位置
            next_value = sequence[-1] // 2
        sequence.append(next_value)
    return sequence


# 生成前10个数字的序列
n = 10
result = generate_sequence(n)
print(result)


def find_next_number(sequence):
    # 检查是否有足够的数字来推测规律
    if len(sequence) < 2:
        return "序列太短，无法推测规律"

    # 定义一个函数来找到下一个数字的规律
    def next_number(seq):
        length = len(seq)

        # 规则1：奇数位的数字（1, 3, 5, ...）
        if length % 2 == 1:
            if seq[-1] % 2 == 0:
                return seq[-1] + 4
            else:
                return seq[-1] + 1
        # 规则2：偶数位的数字（2, 4, 6, ...）
        else:
            if seq[-1] % 2 == 0:
                return seq[-1] // 2
            else:
                return seq[-1] * 2

    # 找到下一个数字
    return next_number(sequence)


# 给定的序列
sequence = [1, 4, 2, 8, 6, 24, 22]

# 预测下一个数字
next_value = find_next_number(sequence)

print(next_value)

alst = []


def next_num(n):
    global alst
    for i in range(0, n):
        if i == 0:
            alst.append(1)
        elif i > 0 and i % 2 != 0:
            alst.append(alst[i - 1] * 4)
        elif i > 0 and i % 2 == 0:
            alst.append(alst[i - 1] - 2)
    return alst


def next_num_squence(n):
    alst = [1]
    [alst.append(alst[i - 1] * 4) if i % 2 != 0 else alst.append(alst[i - 1] - 2) for i in range(1, n)]
    return alst


print(next_num_squence(10))


def factorial(n: int):
    if n == 1:
        return 1
    return n * factorial(n - 1)


print(factorial(4))


def decimal_to_binary(n):
    if n == 0:
        return ''
    return decimal_to_binary(n // 2) + str(n % 2)


print(decimal_to_binary(18))

str2 = lambda x: x.upper() if len(x) < 6 else x
print(str2("ssfda"))

dddd = lambda x: [d for d in x if d > 20]
print(dddd([1, 2, 3, 4, 5, 55]))

data = {"code": 0,
        "data": [
            {"name": "吕布", "age": 18, "love": [{"2012": "girl", "2014": "boy"}], "hurt": 2322, "login": "2021-04-01"},
            {"name": "鲁班", "age": 28, "hurt": 2023, "login": "2021-04-27"},
            {"name": "貂蝉", "age": 38, "hurt": 3212, "login": "2022-05-11"},
            {"name": "亚瑟", "age": 48, "hurt": None, "login": "2023-04-07"},
            {"name": "李白", "age": None, "hurt": 2645, "login": "2022-07-13"}],
        'count': 5}


def next_num(n):
    return [(x + 1) * (2 * x + 1) for x in range(n)][-1]


print(next_num(10))
# 2. 使用 lambda，输入一个数，判断是否为质数
print("=============")
is_preme = lambda x: False if x <= 1 else all(False if x % c == 0 else True for c in range(2, int(x ** 0.5) + 1))
print(is_preme(6))
# 3. 使用 lambda，提取字符串 'abcd5#aa2dasd@dsd0%' 中的数字，输出整数520
s = "abcd5#aa2dasd@dsd0%"
numbers = lambda x: ''.join([c for c in x if c.isdecimal()])
print(numbers(s))
# 4.使用 lambda，从字典列表中，找出薪资大于 5000 的员工，返回元组列表，如[(名字,年龄)]。
lsit_data = [{'name': '宁采臣', 'age': 25, 'salary': 6000},
             {'name': '聂小倩', 'age': 24, 'salary': 4000},
             {'name': '燕赤霞', 'age': 32, 'salary': None},
             {'name': '姥姥', 'age': 45, 'salary': 8000},
             {'name': '黑山老妖', 'age': 45, 'salary': 8000}]
data_name = lambda x: [(c['name'], c['age']) if c['salary'] and c['salary'] > 5000 else () for c in x]
print(data_name(lsit_data))
# 5. 使用 lambda，计算下面 2 个向量 vector 的内积（位置相乘，再相加）使用map也可以做
prod_sum = lambda x, y: sum([math.prod(c) for c in zip(x, y)])
print(prod_sum([1, 2, 3], [4, 5, 6]))

x = int(input('输入一个数'))
prime = lambda x: x > 1 and all(x % n != 0 for n in range(2, int(x // 2) + 1))
if prime(x) == True:
    print(f'{x}是质数')
else:
    print(f'{x}不是质数')


# 实现汉罗塔

def hanoi(n, source, target, temp, moves):
    if n == 1:
        moves.append((source, target))
        return
    hanoi(n - 1, source, temp, target, moves)
    moves.append((source, target))
    hanoi(n - 1, temp, target, source, moves)


moves = []
hanoi(3, 'A', 'C', 'B', moves)
print(len(moves))
for i, v in enumerate(moves):
    print(i, v)
