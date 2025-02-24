import math

list1 = [1, 2, 3]
list2 = [10, 20, 30]
list3 = [100, 200, 300]

print(list(map(lambda x: sum(x), zip(list1, list2, list3))))

data = {[
    {"name": "吕布", "age": 18, "love": [{"2012": "girl", "2014": "boy"}], "hurt": 2322, "login": "2021-04-01"},
    {"name": "鲁班", "age": 28, "hurt": 2023, "login": "2021-04-27"},
    {"name": "貂蝉", "age": 38, "hurt": 3212, "login": "2022-05-11"},
    {"name": "亚瑟", "age": 48, "hurt": None, "login": "2023-04-07"},
    {"name": "李白", "age": None, "hurt": 2645, "login": "2022-07-13"}],
}


def next_num(n):
    return [(x + 1) * (2 * x + 1) for x in range(n)][-1]


print(next_num(10))
# 2. 使用 lambda，输入一个数，判断是否为质数
print("=============")
is_preme = lambda x: False if x <= 1 else all(False if x % c == 0 else True for c in range(2, int(x ** 0.5) + 1))
print(is_preme(5))
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
# 5. 使用 lambda，计算下面 2 个向量 vector 的内积（位置相乘，再相加）
prod_sum = lambda x, y: sum([math.prod(c) for c in zip(x, y)])
print(prod_sum([1, 2, 3], [4, 5, 6]))
map(lambda d: {**d, 'birth': (2024 - d['age'])}, data)

matrix = [[1, 2, 3], [2, 3, 4]]
map(lambda x: [0.001 if not c else c for c in x], matrix)
