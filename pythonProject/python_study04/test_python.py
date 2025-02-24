# 银行：
# 1. 存取的次数，定义实例属性记录存取款的金额[+,-]  、__len__ 获取
#
# 2. 使用 __call__ 方法，把对象当函数，返回值记录（存取款）统计，存款总额，取款总额
from datetime import time, datetime
import time
from functools import wraps


class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
        self.total = 0
        self.record = []

    def __call__(self, *args, **kwargs):
        self.count += 1
        self.total += sum(args)
        time.sleep(1)
        self.record.append(str(time.time())+'次数:'+str(self.count))
        balance = self.func(*args, **kwargs)
        return balance

    def __iter__(self):
        for gen in self.record:
            yield gen

    def __getitem__(self, item):
        return self.record[item]


class Bank:
    balance = 0

    # def __init__(self, balance):
    #     self.balance = balance

    # @CountCalls
    # def deposit(self, money):
    #     if money > 0:
    #         self.balance += money
    #     return self.balance

    @CountCalls
    def deposit(money):
        if money > 0:
            Bank.balance += money
        return Bank.balance

    # @CountCalls
    # def withdraw(self, money):
    #     if self.balance > money:
    #         self.balance -= money
    #     return self.balance

    @CountCalls
    def withdraw(money):
        if Bank.balance > money:
            Bank.balance -= money
        return Bank.balance


bank = Bank()
bank.deposit(10)
bank.deposit(10)
bank.withdraw(10)
for i in bank.deposit.record:
    print(i)
print(bank.balance, bank.deposit.count, bank.withdraw.count, bank.deposit.total, bank.withdraw.total)
