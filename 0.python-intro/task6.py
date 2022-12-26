def check(s, filename):
    bits = s.split(" ")
    words_dict = {}
    bits = map(str.lower, bits)
    for bit in bits:
        if bit in words_dict:
            words_dict[bit] += 1
        else:
            words_dict[bit] = 1

    with open(filename, "w") as f:
        for pair in sorted(words_dict.items()):
            f.write(str(pair[0]) + " " + str(pair[1]) + '\n')
