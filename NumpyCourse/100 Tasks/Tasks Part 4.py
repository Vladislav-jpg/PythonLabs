import numpy as np
#31
a, b = np.array([1,2,3,4]), np.array([5,6,7,8])
print(a+b, a-b, a*b, a/b)
#32
a = np.array([1, 4, 9, 16, 25])
print(np.sqrt(a), np.square(a))
#33
a = np.array([1, 2, 3])
print(np.exp(a), np.log(a))
#34
np.round([1.23, 4.56, 7.89], 1)
#35
np.array([10, 11, 12, 13]) % 3
#36
np.abs([-5, 3, -2, 8, -1])
#37
a = np.array([0, np.pi/2, np.pi])
print(np.sin(a), np.cos(a))
#38
np.power([1,2,3,4], [4,3,2,1])
#39
a, b = np.array([1,2]), np.array([2,1])
print(a>b, a<b, a==b)
#40
np.clip([-5, 5, 15], 0, 10)
