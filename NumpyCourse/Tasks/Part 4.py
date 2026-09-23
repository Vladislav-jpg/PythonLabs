import numpy as np

a = np.array([1, 2, 3, 4])
print(a + 3, (a + 3).dtype)
print(a - 2, (a - 2).dtype)
print(a * 2, (a * 2).dtype)
print(a / 2, (a / 2).dtype)
print(a ** 2, (a ** 2).dtype)
'''ПРИ ДЕЛЕНИИ a / 2 ПРОИЗОШЛО ПРИВЕДЕНИЕ ТИПА К float64 ТАК КАК В РЕЗУЛЬТАТЕ МОГУТ ПОЛУЧИТЬСЯ ДРОБНЫЕ ЧИСЛА'''


a = np.ones((2, 3))
b = np.full((3, 2), 2)
# print(a * b) # Будет ошибка
print(np.matmul(a, b))
'''ПОЭЛЕМЕНТНОЕ УМНОЖЕНИЕ РАБОТАЕТ С МАССИВАМИ ОДИНАКОВОГО РАЗМЕРА А МАТРИЧНОЕ УМНОЖЕНИЕ ТРЕБУЕТ СОВПАДЕНИЯ КОЛИЧЕСТВА СТОЛБЦОВ ПЕРВОЙ МАТРИЦЫ И СТРОК ВТОРОЙ'''


a = np.array([[1, 2, 3], [4, 5, 6]])
print("sum:", a.sum(), a.sum(axis=0), a.sum(axis=1))
print("mean:", a.mean(), a.mean(axis=0), a.mean(axis=1))
print("min:", a.min(), a.min(axis=0), a.min(axis=1))
print("max:", a.max(), a.max(axis=0), a.max(axis=1))
print("prod:", a.prod(), a.prod(axis=0), a.prod(axis=1))
print("var:", a.var(), a.var(axis=0), a.var(axis=1))
'''axis=0 ЭТО ПЕРВАЯ ОСЬ КОТОРАЯ ИДЕТ ВНИЗ ПО СТРОКАМ axis=1 ЭТО ВТОРАЯ ОСЬ КОТОРАЯ ИДЕТ ПО СТОЛБЦАМ'''


m = np.random.rand(4, 4)
print(np.argmax(m.sum(axis=1)))
print(m.mean(axis=0))
print(m - m.mean())