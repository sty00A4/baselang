UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
func join(list, sep)
    if not isList(list) then error("Type Error", "expected list got " + type(x))
    if not isStr(sep) then error("Type Error", "expected string got " + type(x))
    res = ""; i = 0
    forEvery e of list
        res = res + e + (if i == len(list) - 1 then "" else sep); ++i
    end
    return res
end
func sub(str, start, stop)
    if not isStr(str) then error("Type Error", "expected string got " + type(str))
    if not isNum(start) then error("Type Error", "expected number got " + type(start))
    if not isNum(stop) then error("Type Error", "expected number got " + type(stop))
    new_string = ""
    for i = 0, len(str) do if i >= start and i < stop then new_string = new_string + str#i
    return new_string
end
func startswith(str, suf)
    if not isStr(str) then error("Type Error", "expected string got " + type(str))
    if not isStr(suf) then error("Type Error", "expected string got " + type(suf))
    return suf == sub(str, 0, len(suf))
end
func find(str, x)
    if not isStr(str) then error("Type Error", "expected string got " + type(str))
    if not isStr(x) then error("Type Error", "expected string got " + type(x))
    for i = 0, len(str) do if startswith(sub(str, i, len(str)), x) then return i
    return null
end