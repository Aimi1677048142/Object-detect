class MyIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        self.index = 0  # 每次开始迭代时重置索引
        return self

    def __next__(self):
        if self.index < len(self.data):
            current = self.data[self.index]
            self.index += 1
            return current
        else:
            raise StopIteration

# 使用 MyIterator 类
my_iterator = MyIterator([1, 4, 2, 8, 6])
# for num in my_iterator:
#     print(num)
#
# # 再次迭代
# for num in my_iterator:
#     print(num)

next(my_iterator)
next(my_iterator)
next(my_iterator)
next(my_iterator)
next(my_iterator)
