def list_sort(list):
    for i in range(len(list)-1):
        for j in range(len(list)-1):
            if abs(list[j]) < abs(list[j+1]):
                list[j], list[j+1] = list[j+1], list[j]
    return list

list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,-15]
print(list_sort(list))