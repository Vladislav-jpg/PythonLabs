def change(list):
    if len(list) < 2:
        return list
    first = list[0]
    last = list[len(list) - 1]
    list[len(list) - 1] = first
    list[0] = last
    return list
print(change([1,2,3,4,5]))
print(change([1,2]))
print(change([1]))
