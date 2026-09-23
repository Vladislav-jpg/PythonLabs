import numpy as np
OdnomernyiMassive = np.array([1,2,3,4,5])
print(OdnomernyiMassive.ndim)
print(OdnomernyiMassive.dtype)
print(OdnomernyiMassive.size)
print(OdnomernyiMassive.nbytes)
print(OdnomernyiMassive.shape)

DvumerniyMassive = np.array([[1,2,3,4,5], [1,2,3,4,5]])
print(DvumerniyMassive.ndim)
print(DvumerniyMassive.size)
print(DvumerniyMassive.dtype)
print(DvumerniyMassive.nbytes)
print(DvumerniyMassive.shape)

TrimerniyMassive = np.array([[1, 2, 3, 4,], [1, 2, 3, 4], [1, 2, 3, 4]])
print(TrimerniyMassive.ndim)
print(TrimerniyMassive.size)
print(TrimerniyMassive.dtype)
print(TrimerniyMassive.nbytes)
print(TrimerniyMassive.shape)
'''ПОТОМУ ЧТО У ОДНОМЕРНОГО МАССИВА ОДНА ОСЬ РАЗМЕРНОТСИ, ФОМА (1.3) ПОКАЗЫВАЕТ ОДНУ СТРОКА И 3 СТОЛБА'''




arr_int16 = np.zeros(1000, dtype='int16')
arr_int32 = np.zeros(1000, dtype='int32')
arr_float64 = np.zeros(1000, dtype='float64')
print(f"nbytes int16: {arr_int16.nbytes}")
print(f"nbytes int32: {arr_int32.nbytes}")
print(f"nbytes float64: {arr_float64.nbytes}")

print(np.int16(32000) + np.int16(768))
#Потому что байт не хватает для такого значение и ануляюца байты и получаеца минус(отсчет с минимального начинается)

a = np.int8(127)
print(np.int8(a + 1))

