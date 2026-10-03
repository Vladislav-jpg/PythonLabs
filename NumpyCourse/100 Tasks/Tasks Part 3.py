import numpy as np

#21
massive = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(massive[2:6])
#22
print(massive[::-2])
#23
matrix = np.arange(16).reshape(4, 4)
print(matrix[1, :])
print(matrix[:, 2])
#24
massive[massive<5] = 0
#25
array = np.array([0, 3, 0, 7, 0, 9])
print(np.nonzero(massive))
#26
rand_array = np.random.randint(0, 100, 20)
print(rand_array[rand_array %2 == 0])
#27
negative_array = np.array([-1, 2, -3, 4, -5, 6])
print(np.where(negative_array < 0, 0, negative_array))
#28
ma5x5 = np.arange(25).reshape(5, 5)
print(ma5x5[:3, -2:])
#29
array_fi = np.arange(10)
print(array_fi[[1, 3, 5, 7]])
#30
arr = np.arange(10)
print(arr[(arr > 2) & (arr < 8)])