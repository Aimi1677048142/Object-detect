# print("hello world")
s = "MCMXCIV"
result_num = 0
dict1 = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
for i in range(len(s)):
    if s[i] == 'I' and i < (len(s) - 1) and (s[i + 1] == 'V' or s[i + 1] == 'X'):
        result_num -= 1
    elif s[i] == 'X' and i < (len(s) - 1) and (s[i + 1] == 'L' or s[i + 1] == 'C'):
        result_num -= 10
    elif s[i] == 'C' and i < (len(s) - 1) and (s[i + 1] == 'D' or s[i + 1] == 'M'):
        result_num -= 100
    else:
        result_num += dict1.get(s[i])

strs = ["flower", "flow", "flight"]
result = ''
for tmp in zip(*strs):
    tmp_set = set(tmp)
    if len(tmp_set) == 1:
        result += tmp_set.pop()
    else:
        break

while (True):
    num = int(input("请输入数字："))
    if num == 520:
        print("i love you")
        break
    else:
        print("你还是个好人")
