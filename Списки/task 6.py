def all_eq(list):
    strList = []
    for item in list:
        strList.append(str(item))

    max_lenght = len(strList[0])
    for item in strList:
        if len(item) > max_lenght:
            max_lenght = len(item)
    result = []
    for item in strList:
        needs_undercover =  max_lenght - len(item)
        undercover = item + "_" * needs_undercover
        result.append(undercover)
    return result

print(all_eq([1,2,3,4,542]))