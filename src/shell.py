import baselang
from sys import argv
argv = [argv[0]]
while True:
    line = input("by > ")
    if len(line) == 0: continue
    res, error = baselang.run("<stdin>", line)
    if error:
        print(error.as_string())
        continue
    elif res:
        if len(res.elements) > 1: print(repr(res))
        else: print(repr(res.elements[0]))
