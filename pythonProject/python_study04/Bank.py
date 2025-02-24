from types import MethodType


class Bank:
    bank_name = '工商'
    bank_address = '深圳龙岗'
    bank_phone = '12345678'

    def __init__(self, account, balance, password):
        self.account = account
        self.balance = balance
        self.password = password

    def deposit(self, money):
        if money > 0:
            self.balance += money
        return self.balance

    def withdraw(self, money):
        if self.balance > money:
            self.balance -= money
        return self.balance

    @classmethod
    def mod_phone(cls, new_phone):
        cls.bank_phone = new_phone
        return cls.bank_phone

    @staticmethod
    def calc(balance):
        return balance


account = Bank('工商', 100, '123456')
print(account.deposit(10))


# print(account.withdraw(100))
# print(account.calc(account.balance))
# print(account.mod_phone('33333333'))
# print(account.bank_phone)


def deposit(self, money):
    if money > 100:
        self.balance += money
    return self.balance


Bank.deposit = deposit
print(account.deposit(200))  # 修正后的实例方法调用


# 动态绑定类方法
def mod_phone1(cls, new_phone):
    cls.bank_phone = new_phone
    return cls.bank_phone


Bank.mod_phone = classmethod(mod_phone1)

print(Bank.mod_phone('12345'))  # 修正后的类方法调用
print(Bank.bank_phone)


def fib(n):
    x, y = 1, 1
    for _ in range(n):
        yield x
        x, y = y, x + y


for i in fib(10):
    print(i, end=' ')

