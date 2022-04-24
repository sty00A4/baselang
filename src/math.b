PI = 3.141592653589793
func floor(x)
    if not isNum(x) then error("Type Error", "expected number got " + type(x))
    return x // 1
end
func ceil(x)
    if not isNum(x) then error("Type Error", "expected number got " + type(x))
    return if x - x // 1 == 0 then x else x // 1 + 1
end
func round(x)
    if not isNum(x) then error("Type Error", "expected number got " + type(x))
    return if x - x // 1 >= 0.5 then ceil(x) else floor(x)
end
func abs(x)
    if not isNum(x) then error("Type Error", "expected number got " + type(x))
    return if x < 0 then -x else x
end