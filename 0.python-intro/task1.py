def hello(a=None):
    if a == None or a == "":
        return "Hello!"
    else:
        return f"Hello, {a}!"
