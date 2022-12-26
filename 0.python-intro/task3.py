def longestCommonPrefix(a):
    size = len(a)
    a = list(map(str.lstrip, a))
    a.sort()
    end = min(len(a[0]), len(a[size - 1]))
    i = 0
    while i < end and a[0][i] == a[size - 1][i]:
        i += 1
    pre = a[0][0:i]
    return pre
