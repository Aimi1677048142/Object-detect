class SchoolMember:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def tell(self):
        # 打印个人信息
        return 'Name:"{}" Age:"{}"'.format(self.name, self.age)


class Student(SchoolMember):
    def __init__(self, name, age, score):
        super().__init__(name, age)
        self.score = score

    def tell(self):
        # super.tell()
        return super().tell() + ' score:{}'.format(self.score)


class Teacher(SchoolMember):
    def __init__(self, name, age, salary):
        super().__init__(name, age)
        self.salary = salary

    def tell(self):
        super().tell()
        return super().tell() + ' salary:{}'.format(self.salary)


teacher = Teacher('校长', 45, 70000)
student = Student("小明", 16, 90)
print(teacher.tell())
print(student.tell())
