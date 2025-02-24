class CountCalls:
    def __init__(self, func):
        self.func = func
        self.num_calls = 0

    def __call__(self, *args, **kwargs):
        self.num_calls += 1
        return self.func(*args, **kwargs)


@CountCalls
def v(n):
    if n in (0, 1):
        return n
    return v(n - 1) + v(n - 2)

print(v(10), v.num_calls)


def repeat(num_times):
    def decorator_repeat(func):
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                func(*args, **kwargs)

        return wrapper

    return decorator_repeat


@repeat(num_times=3)
def say_hello(name):
    print(f"Hello, {name}!")


say_hello("Alice")


def count_call(fuc):
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return fuc(*args, **kwargs)

    wrapper.call_count = 0

    return wrapper


@count_call
def fibonacci(n):
    if n in (0, 1):
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


print(fibonacci(10), fibonacci.call_count)
