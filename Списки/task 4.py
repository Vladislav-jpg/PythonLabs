def unless(list) -> float:
    maxNum = list[0]
    for i in range(len(list)):
        if list[i] > maxNum:
            maxNum = list[i]

    return float(float(maxNum)/len(list))

print(unless([1,2,3,4,10]))