def bank_account():
    balance = 100
    deposit_count = 0
    withdraw_count = 0

    def deposit(money):
        nonlocal balance, deposit_count
        if money > 0:
            balance += money
            deposit_count += 1
        return balance, deposit_count

    def withdraw(money):
        nonlocal balance, withdraw_count
        if balance > money:
            balance -= money
            withdraw_count += 1
        return balance, withdraw_count

    return deposit, withdraw


deposit1, withdraw1 = bank_account()
print(deposit1(10), withdraw1(100), deposit1(10))
