import basepy
from sys import argv
argv = [argv[0]]
while True:
    res, error = basepy.run("<stdin>", input("by > "))
    if error:
        print(error.as_string())
        continue
    print(res)