from math import sqrt


def is_prime(num) -> bool:
    for i in range(2, int(sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


def primes():
    cur = 1
    while True:
        cur += 1
        if is_prime(cur):
            yield cur
