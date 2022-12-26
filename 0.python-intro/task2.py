def is_palindrome(x):
    reverse = 0
    temp = x
    while x > 0:
        dig = x % 10
        reverse = reverse * 10 + dig
        x = x // 10
    if temp == reverse:
        return "YES"
    else:
        return "NO"
