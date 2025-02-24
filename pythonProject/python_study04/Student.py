class Student:
    number = 0

    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.__score = score
        Student.number += 1

    # 定义打印学生信息的方法
    @property  # 不用括号也可以调用
    def show(self):
        return "Name: {}. Score: {}".format(self.name, self.__score)

    def __str__(self):
        return f"Student(name={self.name},age={self.age},score={self.__score})"

    @classmethod
    def total(cls):
        return f"{cls.number}"

    def __get_score(self):
        return f'{self.__score}'

    @staticmethod
    def func1():
        return "this is static function!"


student = Student('雄安', 24, 13)
student1 = Student('雄安1', 24, 44)
student.high = 180
print(student.high)
print(student.age)
print(student._Student__get_score())
# print(student._score)

x = 10
y = x
print(x)
x = 20
print(y)

poo = [(1, 2), (12, 4), (4, 6), (5, 0)]
poo.sort(key=lambda x: x[1])
print(poo)
map()
