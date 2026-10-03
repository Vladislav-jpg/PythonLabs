import numpy as np
#Первое
print(np.version.__version__)
#Второе
OdnomernyMassive = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#Трерте
Massive_Null = np.zeros((3,3))
Massive_One = np.ones((3,3))
#Четвертое
Seven_Mass = np.full(10, 7)
#Пятое
diagonalnaya_Matrica = np.eye(4)
#Шестое
massive_20 = np.linspace(0, 1, 20)
#Седьмое
massive_ot_10_do_50 = np.arange(10, 50, 5)
#Восьмое
massive_5x5_rand = np.random.rand(5, 5)
#Девятое
Massive42 = np.array([1,2,3])
print(Massive42.dtype)
Massive42.dtype = 'float32'
print(Massive42.dtype)
#Десятое
memory = np.zeros(1000, dtype=np.int64)
print(memory.nbytes)



