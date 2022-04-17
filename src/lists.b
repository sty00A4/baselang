func remove(list, x)
    if not isList(list) then error("Type Error", "expected list got " + type(list))
    var new_list = []
    forEvery e of list do if e != x then var new_list = new_list + e
    return new_list
end
func filter(a, b)
    if not isList(a) then error("Type Error", "expected list got " + type(a))
    if not isList(b) then error("Type Error", "expected list got " + type(b))
    var filtered = []
    forEvery e of listA do if not (e in listB) then var filtered = filtered + e
    return filtered
end
func onlyNum(list)
    if not isList(list) then error("Type Error", "expected list got " + type(list))
    var cond = true
    forEvery e of list
        var cond = isNum(e)
        if not cond then return false
    end
    return true
end
func onlyStr(list)
    if not isList(list) then error("Type Error", "expected list got " + type(list))
    var cond = true
    forEvery e of list
        var cond = isStr(e)
        if not cond then return false
    end
    return true
end
func onlyBool(list)
    if not isList(list) then error("Type Error", "expected list got " + type(list))
    var cond = true
    forEvery e of list
        var cond = isBool(e)
        if not cond then return false
    end
    return true
end
func onlyList(list)
    if not isList(list) then error("Type Error", "expected list got " + type(list))
    var cond = true
    forEvery e of list
        var cond = isList(e)
        if not cond then return false
    end
    return true
end