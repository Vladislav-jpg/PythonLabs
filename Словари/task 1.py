def toDict(list):
    dict = {}
    for i in range(len(list)):
        dict[list[i]] = list[i]
    return dict

print(toDict([1, 2, 3, 4, 5]))
