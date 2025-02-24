import decimal

print(decimal.Decimal('0.1') + decimal.Decimal('0.2'))
print(decimal.Decimal('0.1') + decimal.Decimal('0.2'))
print("你好")

print("hello python!", file=open(r'../python_study04/test.txt', 'w'))

print(-7 // 3)

x = 10
y = x

print(id(x))
print(id(y))

x = 20
print(id(y), y)

nums = [1, 2, 3, 4, 3]
nums.sort()
print(nums)
print(nums.index(2))
nums1 = [x for x in range(10)]

print(nums1)
for i in range(len(nums)):
    print(i)

str1 = "jdfiaf"
print("输出数组 %s" % nums)
len(str1)
print('%10s' % str1)
print('%-10s' % str1)

print('12345\b1234')

print(format('123', '>20'))
print(format('123', '^20'))
print(format('123', '<20'))
dict1 = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

str12 = 'python'
print(str12[::-1])
print(str12[::-2])
print(str12[::2])

print(str12.find('n', 4, len(str12)))
print('1\n2\r3\n4'.splitlines())

s = '   www   python  org.cn  '
print(s.replace(' ', ','))
print(s.lstrip('w p'))
print(s.rstrip('w p'))

print(s.index('p'))
print('我'.encode('utf-8'))
