def process(l):
    return sorted(set(a*a for b in l for a in b))[::-1]
