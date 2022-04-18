func remove(list, x)
    if not isList(list) then error("Type Error", "expected list got " + type(list))
    new_list = []
    forEvery e of list do if e != x then new_list = new_list + e
    return new_list
end
func filter(a, b)
    if not isList(a) then error("Type Error", "expected list got " + type(a))
    if not isList(b) then error("Type Error", "expected list got " + type(b))
    filtered = []
    forEvery e of listA do if not (e in listB) then filtered = filtered + e
    return filtered
end
func onlyNum(list)
    if not isList(list) then error("Type Error", "expected list got " + type(list))
    cond = true
    forEvery e of list
        cond = isNum(e)
        if not cond then return false
    end
    return true
end
func onlyStr(list)
    if not isList(list) then error("Type Error", "expected list got " + type(list))
    cond = true
    forEvery e of list
        cond = isStr(e)
        if not cond then return false
    end
    return true
end
func onlyBool(list)
    if not isList(list) then error("Type Error", "expected list got " + type(list))
    cond = true
    forEvery e of list
        cond = isBool(e)
        if not cond then return false
    end
    return true
end
func onlyList(list)
    if not isList(list) then error("Type Error", "expected list got " + type(list))
    cond = true
    forEvery e of list
        cond = isList(e)
        if not cond then return false
    end
    return true
end